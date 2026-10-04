# global_priority_planner.py (IMPROVED VERSION)
"""
Global planner with LOCAL MAXIMA extraction.
Finds multiple peaks even within same connected region.
Priority = coverage / distance (higher is better).

Uses the EXACT coordinate transformation as autonomous_explorer5.py
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from math import pi
import cv2
from scipy.ndimage import maximum_filter
from sklearn.cluster import KMeans

@dataclass
class CandidateGoal:
    """A candidate goal in world coordinates"""
    world_pos: Tuple[float, float]  # (x, y) in world pixels
    coverage_value: float  # Coverage prediction value (0-1 range)
    timestamp: float  # When discovered
    visited: bool = False
    peak_id: int = -1  # Which local maximum this is


class GlobalPriorityPlanner:
    """
    Local maxima-based global planner.
    - Finds ALL local maxima in coverage prediction
    - Selects top K by value with spatial clustering
    - Stores candidates in world coordinates
    - Priority = coverage / distance (higher = better)
    - Unlimited memory (keeps all candidates ever found)

    KEY DIFFERENCE FROM REGION-BASED:
    - Region-based finds one point per connected region
    - This finds MULTIPLE peaks even within same region!
    - Better for detecting multiple local maxima
    """

    def __init__(self, coverage_threshold: float = 0.0001,
                 top_k_maxima: int = 5,
                 min_maxima_distance_m: float = 2.0,
                 local_maxima_radius_px: int = 15,
                 resolution_m_per_pixel: float = 0.05):
        """
        Args:
            coverage_threshold: Min coverage value to consider (e.g., 0.0001)
            top_k_maxima: Number of local maxima to extract (default 5)
            min_maxima_distance_m: Minimum distance between local maxima in meters
            local_maxima_radius_px: Window size for non-maximum suppression (in pixels)
            resolution_m_per_pixel: Map resolution (meters per pixel) for converting distance
        """
        self.coverage_threshold = coverage_threshold
        self.top_k_maxima = top_k_maxima
        self.min_maxima_distance_m = min_maxima_distance_m
        self.local_maxima_radius_px = local_maxima_radius_px
        self.resolution_m_per_pixel = resolution_m_per_pixel

        # Convert distance from meters to pixels
        self.min_maxima_distance_px = int(min_maxima_distance_m / resolution_m_per_pixel)

        self.candidates = []  # List of all discovered candidates
        self.current_target = None  # Currently selected target

        print(f"\n{'=' * 60}")
        print(f"GLOBAL PRIORITY PLANNER (Local Maxima Based)")
        print(f"{'=' * 60}")
        print(f"Coverage threshold: {self.coverage_threshold}")
        print(f"Top K local maxima: {self.top_k_maxima}")
        print(f"Min maxima distance: {self.min_maxima_distance_m}m ({self.min_maxima_distance_px}px)")
        print(f"Local maxima radius (NMS): {self.local_maxima_radius_px}px")
        print(f"Map resolution: {self.resolution_m_per_pixel}m/px")
        print(f"Memory: Unlimited")
        print(f"Priority metric: coverage / distance")
        print(f"{'=' * 60}\n")

    def _find_local_maxima(self, heatmap: np.ndarray) -> List[Tuple[float, float, float]]:
        """
        Find all local maxima in heatmap using non-maximum suppression.
        Returns maxima even if they're in same connected region!

        Returns:
            List of (x, y, value) tuples sorted by value (descending)
        """
        # Ensure heatmap is float
        heatmap = heatmap.astype(np.float32)

        # Find local maxima using maximum filter
        # For each pixel, check if it's the max in a neighborhood
        local_max_filter = maximum_filter(heatmap, size=self.local_maxima_radius_px)
        local_maxima = (heatmap == local_max_filter)

        # Apply threshold
        local_maxima[heatmap < self.coverage_threshold] = False

        # Get coordinates of all local maxima
        coords = np.where(local_maxima)

        if len(coords[0]) == 0:
            return []

        # Get values at these coordinates
        values = heatmap[coords]

        # Sort by value (descending)
        sorted_indices = np.argsort(values)[::-1]

        # Return as list of (x, y, value) tuples
        maxima = []
        for idx in sorted_indices:
            y, x = coords[0][idx], coords[1][idx]
            maxima.append((float(x), float(y), float(values[idx])))

        return maxima

    def _select_top_k_maxima_spatially_clustered(self,
                                                 maxima: List[Tuple[float, float, float]]) -> List[
        Tuple[float, float, float]]:
        """
        Select top K maxima with spatial clustering to avoid redundancy.

        Args:
            maxima: List of (x, y, value) tuples sorted by value

        Returns:
            List of selected (x, y, value) tuples, spatially separated
        """
        if len(maxima) == 0:
            return []

        selected = []

        for max_x, max_y, max_val in maxima:
            if len(selected) >= self.top_k_maxima:
                break

            # Check if this maximum is too close to already selected ones
            too_close = False
            for sel_x, sel_y, sel_val in selected:
                distance = np.sqrt((max_x - sel_x) ** 2 + (max_y - sel_y) ** 2)

                if distance < self.min_maxima_distance_px:
                    too_close = True
                    break

            if not too_close:
                selected.append((max_x, max_y, max_val))

        return selected

    def add_candidates_from_heatmap(self, heatmap: np.ndarray,
                                    robot_pos_world: Tuple[float, float],
                                    origin_pose: Tuple[float, float, float],
                                    window_px: int,
                                    timestamp: float,
                                    resolution: float = 0.05):
        """
        Extract local hotspots from heatmap using CONTOUR-BASED CENTROIDS.
        This captures the "centers of mass" of high-value regions.
        """
        # 1. Thresholding to find high-value regions
        # Use a relative threshold (e.g., 70% of max) or absolute (e.g., 0.5)
        max_val = heatmap.max()
        if max_val < self.coverage_threshold:
            return

        # Adaptive threshold: 70% of peak, but at least 0.1
        threshold = max(0.1, max_val * 0.7)
        binary_map = (heatmap > threshold).astype(np.uint8)

        # 2. Find Contours
        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        new_candidates = []

        print(f"PLANNER: Found {len(contours)} hotspot regions (thresh={threshold:.3f})")

        for idx, cnt in enumerate(contours):
            # Filter small noise
            if cv2.contourArea(cnt) < 5: continue

            # 3. Compute Centroid
            M = cv2.moments(cnt)
            if M["m00"] == 0: continue

            # Centroid in local image coordinates
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # 4. Get value at centroid (or max value within contour)
            # Option A: Value at centroid
            # val = heatmap[cy, cx]

            # Option B: Max value within this specific contour region
            mask = np.zeros_like(heatmap, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 1, -1)
            # Mask the heatmap
            masked_heatmap = heatmap * mask
            # Find max location in this region
            min_v, max_v, min_loc, max_loc = cv2.minMaxLoc(masked_heatmap)

            # Use max location instead of centroid for better precision?
            # User asked for "centroids", but max point is usually safer for navigation.
            # Let's stick to Centroid as requested, but assign it the max value of the region.
            # Actually, centroid is better for "center of mass".

            # Let's use the centroid coordinate, but the max value of the region as score.
            coverage_val = max_v

            # Transform to world coordinates
            world_pos = self._transform_image_goal_to_world(
                (cx, cy),
                origin_pose,
                window_px
            )

            # Create candidate
            candidate = CandidateGoal(
                world_pos=world_pos,
                coverage_value=float(coverage_val),
                timestamp=timestamp,
                visited=False,
                peak_id=idx
            )
            new_candidates.append(candidate)

            print(f"PLANNER:   Hotspot {idx}: Local({cx},{cy}) -> World({world_pos[0]:.1f},{world_pos[1]:.1f}) Val={coverage_val:.4f}")

        # Add to memory
        self.candidates.extend(new_candidates)
        print(f"PLANNER: Added {len(new_candidates)} new candidates. Total: {len(self.candidates)}")

    def _transform_image_goal_to_world(self, goal_px: Tuple[int, int],
                                       origin_pose: Tuple[float, float, float],
                                       image_size: int) -> Tuple[float, float]:
        """
        Transform goal from local image frame to world frame.
        MATCHES the transformation in autonomous_explorer5.py exactly.
        """
        center_x, center_y = image_size // 2, image_size // 2
        origin_x, origin_y, origin_theta = origin_pose

        # Convert theta to degrees and adjust
        rot_angle_deg = 90.0 - (origin_theta * 180.0 / pi)
        rot_angle_rad_inv = rot_angle_deg * pi / 180.0

        # Compute offsets from center
        dx_img = goal_px[0] - center_x
        dy_img = goal_px[1] - center_y

        # Apply rotation
        cos_rot = np.cos(rot_angle_rad_inv)
        sin_rot = np.sin(rot_angle_rad_inv)

        dx_world = dx_img * cos_rot - dy_img * sin_rot
        dy_world = dx_img * sin_rot + dy_img * cos_rot

        world_x = origin_x + dx_world
        world_y = origin_y + dy_world

        return (world_x, world_y)

    def select_next_target(self, robot_pos_world: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """
        Select next target based on priority = Coverage + Cluster - Distance.
        Implements the "Sweeping" strategy.
        """
        # Filter unvisited candidates
        unvisited = [c for c in self.candidates if not c.visited]

        if len(unvisited) == 0:
            print("PLANNER: No unvisited candidates!")
            return None

        # Pre-calculate candidate locations for clustering
        candidate_locations = [c.world_pos for c in unvisited]
        cluster_radius_m = 5.0

        best_candidate = None
        best_priority_value = -np.inf

        for candidate in unvisited:
            # 1. Distance Cost
            dx = candidate.world_pos[0] - robot_pos_world[0]
            dy = candidate.world_pos[1] - robot_pos_world[1]
            dist_px = np.sqrt(dx * dx + dy * dy)
            dist_m = dist_px * self.resolution_m_per_pixel

            # 2. Cluster Bonus (Count neighbors)
            cluster_score = 0.0
            for other_loc in candidate_locations:
                if other_loc == candidate.world_pos: continue
                # Euclidean distance between candidates
                d_other = np.hypot(candidate.world_pos[0] - other_loc[0], candidate.world_pos[1] - other_loc[1])
                if (d_other * self.resolution_m_per_pixel) < cluster_radius_m:
                    cluster_score += 1.0

            # 3. Weighted Utility Function
            # Efficiency = (Coverage * 1.0) + (Cluster * 0.15) - (Distance * 0.05)
            priority_value = (candidate.coverage_value * 1.0) + \
                             (cluster_score * 0.15) - \
                             (dist_m * 0.05)

            if priority_value > best_priority_value:
                best_priority_value = priority_value
                best_candidate = candidate

        if best_candidate is None:
            return None

        # Mark as visited
        best_candidate.visited = True
        self.current_target = best_candidate.world_pos

        # Compute distance for logging
        dx = best_candidate.world_pos[0] - robot_pos_world[0]
        dy = best_candidate.world_pos[1] - robot_pos_world[1]
        distance = np.sqrt(dx * dx + dy * dy)

        print(f"PLANNER: Selected target at {best_candidate.world_pos}")
        print(f"         Peak ID: {best_candidate.peak_id}")
        print(f"         Coverage: {best_candidate.coverage_value:.6f}")
        print(f"         Cluster Score: {cluster_score:.1f}")
        print(f"         Distance: {distance:.1f} px ({distance * self.resolution_m_per_pixel:.1f} m)")
        print(f"         Priority: {best_priority_value:.6f}")

        return best_candidate.world_pos

    def get_all_candidates(self) -> List[dict]:
        """Get all candidates for visualization."""
        result = []
        for candidate in self.candidates:
            result.append({
                'pos': candidate.world_pos,
                'coverage': candidate.coverage_value,
                'visited': candidate.visited,
                'is_target': (candidate.world_pos == self.current_target) if self.current_target else False,
                'peak_id': candidate.peak_id
            })
        return result

    def get_stats(self) -> dict:
        """Get statistics about candidates."""
        total = len(self.candidates)
        visited = len([c for c in self.candidates if c.visited])
        unvisited = total - visited

        if self.candidates:
            avg_coverage = np.mean([c.coverage_value for c in self.candidates])
            max_coverage = max([c.coverage_value for c in self.candidates])
        else:
            avg_coverage = 0.0
            max_coverage = 0.0

        return {
            'total_candidates': total,
            'visited': visited,
            'unvisited': unvisited,
            'avg_coverage': avg_coverage,
            'max_coverage': max_coverage
        }

    def clear_memory(self):
        """Clear all stored candidates."""
        self.candidates.clear()
        self.current_target = None
        print("PLANNER: Memory cleared")


    def find_global_frontiers(self, occ: np.ndarray, explored: np.ndarray,
                             robot_pos_world: Tuple[float, float],
                             nudge_dist_px: int = 30) -> List[dict]:
        """
        Identifies frontiers with SIZE FILTERING only.
        Enabled: Noise Filter (Min Size = 5)
        Disabled: Safety Filter (Wall Proximity) - to prevent deleting doorways
        """
        H, W = occ.shape

        # 1. Define masks
        free_mask = (explored == 1) & (occ == 0)
        unknown_mask = (explored == 0)

        # 2. Find frontiers (Free edges touching Unknown)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated_free = cv2.dilate(free_mask.astype(np.uint8), kernel)
        frontier_mask = (dilated_free == 1) & (unknown_mask == 1)

        # --- FILTER 1 DISABLED: Safety Mask ---
        # We perform no wall proximity filtering to ensure we don't delete narrow doors.
        safe_frontier_mask = frontier_mask

        # 4. Find Contours
        contours, _ = cv2.findContours(safe_frontier_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []

        # --- FILTER 2 ENABLED: Minimum Size ---
        min_contour_size = 30  # Ignore frontiers smaller than 30 pixels

        for cnt in contours:
            # Apply Size Filter
            if len(cnt) < min_contour_size: continue

            # Centroid
            M = cv2.moments(cnt)
            if M["m00"] == 0: continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Vector Calculation (Push centroid slightly into free space)
            y_int, x_int = int(cy), int(cx)
            win = 10
            y1, y2 = max(0, y_int-win), min(H, y_int+win)
            x1, x2 = max(0, x_int-win), min(W, x_int+win)

            vec_x, vec_y = 0, 0
            if y2 > y1 and x2 > x1:
                local_free = free_mask[y1:y2, x1:x2].astype(np.float32)
                M_local = cv2.moments(local_free)
                if M_local["m00"] != 0:
                    cX_local = M_local["m10"] / M_local["m00"] + x1
                    cY_local = M_local["m01"] / M_local["m00"] + y1
                    vec_x = cx - cX_local
                    vec_y = cy - cY_local

            if vec_x == 0 and vec_y == 0:
                vec_x = cx - robot_pos_world[0]
                vec_y = cy - robot_pos_world[1]

            mag = np.sqrt(vec_x**2 + vec_y**2)
            if mag == 0: continue
            vx, vy = vec_x / mag, vec_y / mag

            # Nudge
            nudge_x = cx - (vx * nudge_dist_px)
            nudge_y = cy - (vy * nudge_dist_px)
            angle = np.arctan2(-vy, vx)

            # Distance
            dist = np.hypot(cx - robot_pos_world[0], cy - robot_pos_world[1])

            candidates.append({
                'pose': (nudge_x, nudge_y, angle),
                'contour': cnt,
                'dist': dist
            })

        # SORTING: Prioritize closest frontiers
        candidates.sort(key=lambda x: x['dist'])

        return candidates
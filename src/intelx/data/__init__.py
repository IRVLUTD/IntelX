"""Generating the training set, and reading it back.

`driver` samples spawns and runs the shows that produce coverage data for one
map. `ring_collector` holds the persistent GPU workers they both drive.
`dataset` is the chunked reader training uses. `collector` is the CSV logger.

`intelx.cli.collect_data` is the entry point that runs this across every map.
"""

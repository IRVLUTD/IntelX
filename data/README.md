# Data

Two separate things live here. `data/maps` holds the floor plans, which are
inputs. `data/coverage` holds the generated training set, which the collectors
write and training reads. They are separate roots because a coverage chunk is
not a floor plan and the two are usually on different disks.

The test, out-of-distribution and real maps are at
[Maps](https://utdallas.box.com/s/stgu95mdcf3p7tgp5z5a53pcxiuwrplu), in its
`maps/` folder. The 18 training maps and the coverage dataset (about 1 TB) are
not public yet; `collect_data` regenerates the dataset from the training maps.
Three of the maps are included, one of each kind.

| Map | Kind |
|---|---|
| `data/maps/test/small_office3.pgm` | a held-out KTH floor plan |
| `data/maps/ood/aws_maps/hospital.pgm` | an out-of-distribution AWS RoboMaker world |
| `data/maps/real/ecss_corridor.pgm` | the real corridor, from a SLAM run |

Point either root elsewhere with an environment variable and change nothing in
the source.

```bash
export INTELX_MAPS_ROOT=/path/to/your/maps
export INTELX_DATASET_ROOT=/path/to/your/coverage/data
```

## Maps

Occupancy maps are PGM files at 0.05 m per pixel, sealed so that walls are
watertight. `intelx.paths.map_path` resolves every one of them under
`data/maps`, so a path in the code is always relative.

```
data/maps/
  train/                       the 18 training maps, 17 KTH floor plans and
                               the AWS small house
  test/                        the three held-out KTH maps
  ood/
    house_expo/{small,medium,large}/
    msd/{small,medium,large}/  Modified Swiss Dwellings
    aws_maps/                  AWS RoboMaker worlds
  real/                        the real corridor scan
```

### Sources

The maps are standardised from four public datasets, each under its own
licence: KTH floor plans, HouseExpo, Modified Swiss Dwellings and the AWS
RoboMaker worlds. The `maps/` folder of the download unpacks into the tree
above.

`data/train_metadata.csv` lists the 18 training maps with the shows each
receives, Table 1 of the paper, allocated in proportion to the complexity
score of Eq. (1). `intelx.cli.collect_data` reads it.

## The Coverage Dataset

`intelx.paths.dataset_path` resolves everything the collectors write and
everything training reads, under `data/coverage`.

```
data/coverage/
  coverage_data_main_3/    what intelx.cli.collect_data writes, per map
  train_chunks/            the chunked form intelx.cli.train reads
  dense_data/<map>/        what a single-map run of intelx.data.driver writes
```

It is not public yet, so generate it.
`python -m intelx.cli.collect_data` runs the collection across every map in
`INTELX_MAPS_ROOT`, and it is a long multi-GPU job, so read the constants at the
top of that file first. `<CHUNKING_SCRIPT>` then packs what it writes into
`train_chunks/`.

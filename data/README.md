# Data

Two separate things live here. `data/maps` holds the floor plans, which are
inputs. `data/coverage` holds the generated training set, which the collectors
write and training reads. They are separate roots because a coverage chunk is
not a floor plan and the two are usually on different disks.

Only three sample maps are committed, one of each kind, listed in the README.
The training set is not. Point either root elsewhere with an environment
variable and change nothing in the source.

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
  train_standardized/          the 18 training maps, 17 KTH floor plans and
                               the AWS small house
  test_standardized/           the three held-out KTH maps
  ood_test_standardized/
    house_expo/{small,medium,large}/
    msd/{small,medium,large}/  Modified Swiss Dwellings
    aws_maps/                  AWS RoboMaker worlds
  msd_final/{small,medium,large}/
  real/                        the real corridor scan
```

### Sources

Every source dataset is public and separately available. KTH floor plans,
HouseExpo, Modified Swiss Dwellings and the AWS RoboMaker worlds each carry
their own licence, which is why the standardised PGMs are not committed here.
Either drop the standardised maps into the tree above, or add the script that
regenerates them from the originals.

`data/train_metadata.csv` lists the 18 training maps with the shows each
receives, Table 1 of the paper, allocated in proportion to the complexity
score of Eq. (1). `intelx.cli.collect_data` reads it.

## The coverage dataset

`intelx.paths.dataset_path` resolves everything the collectors write and
everything training reads, under `data/coverage`.

```
data/coverage/
  coverage_data_main_3/    what intelx.cli.collect_data writes, per map
  train_chunks/            the chunked form intelx.cli.train reads
  dense_data/<map>/        what a single-map run of intelx.data.driver writes
```

Generate it rather than downloading it. `python -m intelx.cli.collect_data`
runs the collection across every map in `INTELX_MAPS_ROOT`, and it is a long
multi-GPU job, so read the constants at the top of that file first.

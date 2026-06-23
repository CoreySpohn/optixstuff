# Changelog

## [1.4.0](https://github.com/CoreySpohn/optixstuff/compare/v1.3.0...v1.4.0) (2026-06-23)


### Miscellaneous Chores

* release 1.4.0 ([39c1d25](https://github.com/CoreySpohn/optixstuff/commit/39c1d25741f3aaa47984d32e0e6c7917933a59ba))

## [1.3.0](https://github.com/CoreySpohn/optixstuff/compare/v1.2.0...v1.3.0) (2026-06-23)


### Features

* Add Zenodo citation ([0028643](https://github.com/CoreySpohn/optixstuff/commit/0028643e1f4079db6543a1fc43ddf353b7df226b))

## [1.2.0](https://github.com/CoreySpohn/optixstuff/compare/v1.1.0...v1.2.0) (2026-06-18)


### Features

* Add reference wavelength for the psflets ([a2fe241](https://github.com/CoreySpohn/optixstuff/commit/a2fe241ee778ef54c4832e68950fe5268f8a8b1c))
* **disperser:** wavelength-dependent throughput via composed optical element ([c902c88](https://github.com/CoreySpohn/optixstuff/commit/c902c88ae08447e5dfbc868cbcc51a254098877e))

## [1.1.0](https://github.com/CoreySpohn/optixstuff/compare/v1.0.2...v1.1.0) (2026-05-30)


### Features

* add AbstractDisperser and LensletDisperser descriptors ([6e9cd21](https://github.com/CoreySpohn/optixstuff/commit/6e9cd210c3b6f9b8af3b4b8a72cc85e412288c39))
* **detector:** add per-pixel noise_variance (shot + dark + CIC + read) ([f983b0d](https://github.com/CoreySpohn/optixstuff/commit/f983b0dc628e6a2ef8f6489a396c18ff2fff173a))
* **optical_path:** add optional disperser descriptor field ([06e486e](https://github.com/CoreySpohn/optixstuff/commit/06e486e31daaa15773d91f2b586ad7ec16359db5))

## [1.0.2](https://github.com/CoreySpohn/optixstuff/compare/v1.0.1...v1.0.2) (2026-05-25)


### Bug Fixes

* Add mermaid rendering to docs ([26e1cf0](https://github.com/CoreySpohn/optixstuff/commit/26e1cf0f8b43fe9ff16fe9c66e7bd877adf9ccd6))

## [1.0.1](https://github.com/CoreySpohn/optixstuff/compare/v1.0.0...v1.0.1) (2026-05-25)


### Bug Fixes

* Add readthedocs yaml file ([4f2380e](https://github.com/CoreySpohn/optixstuff/commit/4f2380e3e14cdecd40059bbec1a1d9d8c5684512))

## [1.0.0](https://github.com/CoreySpohn/optixstuff/compare/v0.2.0...v1.0.0) (2026-05-25)


### ⚠ BREAKING CHANGES

* Docs and refactor to standardize function calls
* SimpleDetector -> IdealDetector (clarifies that the class ignores wavelength-dependent QE and noise contributions); ConstantThroughputElement -> ConstantThroughput and LinearThroughputElement -> LinearThroughput (the Element suffix was redundant with the inheritance from AbstractUniformElement); AbstractScalarOnlyCoronagraph -> AbstractScalarCoronagraph (the Only qualifier was redundant); Exposure -> ExposureConfig (settings-struct role explicit). README ecosystem diagram replaced with Mermaid and sphinxcontrib-mermaid added to the docs build.

### Features

* Docs and refactor to standardize function calls ([5e5ab2c](https://github.com/CoreySpohn/optixstuff/commit/5e5ab2c28ac3d4dc387ba7eee7407740f95648c3))
* rename detector/throughput classes and Exposure for naming consistency ([10beac7](https://github.com/CoreySpohn/optixstuff/commit/10beac7755e38c1743aa91931739453c9a5f50ed))

## [0.2.0](https://github.com/CoreySpohn/optixstuff/compare/v0.1.0...v0.2.0) (2026-05-24)


### Features

* **optixstuff:** Add __repr__ and OpticalPath.from_default_setup ([bef51c0](https://github.com/CoreySpohn/optixstuff/commit/bef51c03f2a5845ffd1db3d613cf41d502ede0fa))
* **optixstuff:** Add readout_source_electrons_thinned for fast detector readout ([f3e7588](https://github.com/CoreySpohn/optixstuff/commit/f3e7588a1253b7cb9ebc30ec7f20f14634656527))


### Bug Fixes

* **optixstuff:** Accept yippy.EqxCoronagraph in OpticalPath.from_default_setup ([fed5b2d](https://github.com/CoreySpohn/optixstuff/commit/fed5b2d7a0e422f4a7738861ba7d60f80db188ba))
* **optixstuff:** Use yippy.fetch_yip in integration test (renamed upstream) ([8261c60](https://github.com/CoreySpohn/optixstuff/commit/8261c60be8da0506fec8f7ad7cc510c0d341287f))

## [0.1.0](https://github.com/CoreySpohn/optixstuff/compare/v0.0.1...v0.1.0) (2026-04-23)


### Features

* **detector:** split add_noise into add_source_electrons + add_noise_electrons ([6f76008](https://github.com/CoreySpohn/optixstuff/commit/6f760089d49d912c74a6c119dad174a37b3264d8))
* **exposure:** adopt Exposure module from coronagraphoto ([21eeb20](https://github.com/CoreySpohn/optixstuff/commit/21eeb20a7cffd06e48b422b3f30e8a93ebd91335))
* **optical_path:** add n_channels and npix_multiplier fields ([93a4b51](https://github.com/CoreySpohn/optixstuff/commit/93a4b5152f06adf7b7abd9a12bd7b6b1ef4ecc31))
* **yippy_coronagraph:** add create_psfs and psf_datacube for image consumers ([b74e724](https://github.com/CoreySpohn/optixstuff/commit/b74e7248ed9c312cbcdf89c12a5af21a15fe5d87))

## 0.0.1 (2026-04-13)


### Features

* implement wavelength-dependent throughput elements and YippyCoronagraph integration ([a273eac](https://github.com/CoreySpohn/optixstuff/commit/a273eac61a97bbff00aee9e147f9b2736ac5e71a))
* initial package scaffold ([909ab74](https://github.com/CoreySpohn/optixstuff/commit/909ab747971deca8c62570620b693361f55d3419))


### Miscellaneous Chores

* release 0.0.1 ([610191e](https://github.com/CoreySpohn/optixstuff/commit/610191ec673ba90d8380aa60fa2a45592cc14334))

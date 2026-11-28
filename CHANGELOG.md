# Changelog

## [7.0.0](https://github.com/CoreySpohn/optixstuff/compare/v6.1.0...v7.0.0) (2026-11-28)


### ⚠ BREAKING CHANGES

* **exposure:** Pointing record replaces Exposure.roll_deg
* **optixstuff:** close grid, spectral, rates, detector and speckle forwards at their working precision
* **optixstuff:** dispatch bands, check ranges and validate PSFlet axes the same way in both precision lanes; clock refusals raise PrecisionError
* **coronagraph:** every coronagraph declares band_averaged; Quadrature.build refuses an observing band wider than 1.25 times a band product's fractional width unless allow_wide_bands is set, and always refuses a band straddling two band products.
* **exposure:** frames_for_time returns n_frames NaN for a non-finite or negative time_s (zero seconds still gives zero frames) and counts the smallest whole number of frames whose product with the frame time covers the request in the working dtype, replacing the relative rounding tolerance; FramePlan raises for a concrete non-finite or non-positive frame_time_s and for a negative or fractional n_frames.
* **performance:** PerformanceCurves gains a uniform_sky field (a SampledProfile in the record's frame, or None), which changes the record's pytree; performance_curves accepts uniform_sky as a profile or a sky Image. A leakage read at a tabulated stellar diameter now returns that row exactly, so a measured zero stays zero instead of the log floor.
* **rates:** Rates refuses a nonscalar component whose rank or leading axes do not match the record (reshape per-band components to (n_band, 1, 1)), the shape is re-checked at total, integrate and the detector conversion, and the detector statistics refuse spectral-density or non-electron records before flooring.
* **speckle:** FocalResponse takes a required keyword-only wavelength_law ("opd", "amplitude" or None for a tabulated response). An amplitude response no longer scales its sensitivity with wavelength; an OPD response keeps the reference_wavelength_nm / wavelength scaling.
* **psflet_pack:** PsfletPack takes a required template_origin (only "centroid" is accepted) and format 2 files without it are refused, so legacy origin-relative templates cannot be labelled format 2. The version tag must be a scalar integer.
* **psflet_pack:** write pack format 2 and refuse format 1 files
* **speckle:** every AbstractSpeckleField and LinearSpeckleField method takes keyword-only elapsed_s, seconds since the field's clock origin, in place of time_d (coefficients(*, elapsed_s), realize(..., elapsed_s=), broadened(..., elapsed_s=)); realize_average takes start_elapsed_s in place of start_time_d; positional coefficients(t) and the time_d keyword raise TypeError. Implementations no longer read epoch_d. Scene days convert through the concrete AbstractSpeckleField.elapsed_s(time_d), which subtracts epoch_d in binary64 and raises ValueError for a time that is not binary64 when epoch_d is nonzero (a float32 JAX array, a traced Python float under x32, or a NumPy float32); pass elapsed_s formed on the host, enable x64, or move the scene epoch so that epoch_d is zero.
* **grids:** CellOverlap.axes and CellOverlap.sparse hold jax.Array (indices int32, weights in the active float dtype) instead of NumPy arrays.
* **detector:** Detector noise fields and _qe are strong arrays of the active dtype (were weak or default-dtype jnp.asarray results).
* **spectral:** Reduction.build refuses a plain HostGrid with invalid knots again; product tables pass ProductEdges.
* **spectral:** the Detector QE curve and PsfletPack wavelengths are static Knots (Detector._qe_curve is now _qe_knots and _qe_values), Exposure drops reads and holds time_d as a float leaf, Quadrature.build refuses pieces whose nodes collide at float32 in either precision, and chained break clusters wider than one tolerance are refused.
* **spectral:** SpectralTable declares knots (a Knots) and a str knot_digest; spectral_breaks_nm is no longer part of the protocol, and a table without static Knots is refused with TypeError instead of being re-hashed from its breaks.
* **spectral:** Transmission.wavelength_nm is a static Knots carried as pytree auxiliary data and transmission is its only leaf; Transmission.low_nm and high_nm are floats; breaks, band edges and table ends within four float32 ULPs (merge_tolerance_nm) are one wavelength in x32 and x64; Knots refuses a grid whose adjacent knots are that close; SpectralTable tables declare a static knots grid and a knot_digest property.
* **spectral:** Quadrature.weights() takes no argument; call weights() in place of weights("response").
* **grids:** on_grid reads nearest, linear and cubic densities as the integral of their zero-extended reconstruction over each pixel instead of a midpoint cell integral then resample, so their values change (a linear density at the native pitch is smoothed by 1/8, 3/4, 1/8 per axis) and, at every pitch, pixels within the kernel reach beyond the table receive light (coverage supports none of them). on_grid of such a density on a rotated or sheared grid now raises NotImplementedError. on_grid and resample (cubic_spline and band_limited) raise NotImplementedError when the target orientation is not the source orientation object and either angle is traced, so an equal-valued orientation built as another object also raises; the band-limited branch stopped with a TracerArrayConversionError before. resample still samples densities on rotated host grids.
* **performance:** PerformanceCurves leakage quantities and raw_contrast have shape (K, S) with stellar_diameters_lod; evaluate takes stellar_diameter_lod for them and the static stellar_diameter_lod field is removed.
* **spectral:** Quadrature holds one weight row and static pieces; field_lod raises the order of pieces the response slides across
* **performance:** PerformanceCurves.supported ends where the samples end, evaluate beyond them is not data
* **grids:** delete Image.outside and the reflect branches of resample; an image is zero past its grid
* **coronagraph:** a scene coronagraph declares the scene_grid its render_scene reads
* **instrument:** add WarmEmission and Instrument.warm_emission for the warm telescope's thermal emission
* **coronagraph:** rename Coronagraph.band to coronagraph_at
* **calculator:** require post_processing in snr and exposure_time
* **coronagraph:** rename MultiBandCoronagraph.bands to coronagraphs
* **calculator:** require background_variance_factor with no default
* **instrument:** delete the unread n_channels and npix_multiplier fields
* **calculator:** move the S/N and exposure-time calculator here as PostProcessing with a background variance factor
* **coronagraph:** trim the slot surface and spell stellar_diameter_lod everywhere
* **instrument:** remove the OpticalPath alias of Instrument
* **coronagraph:** reduce the coronagraph slot to the image capability

### Features

* **calculator:** move the S/N and exposure-time calculator here as PostProcessing with a background variance factor ([933dc7d](https://github.com/CoreySpohn/optixstuff/commit/933dc7d5397272c45f86b14251649dff63d3a42b))
* **calculator:** require background_variance_factor with no default ([3f1fa83](https://github.com/CoreySpohn/optixstuff/commit/3f1fa835f32e27946cfbcbe3d80decc00c6fded5))
* **calculator:** require post_processing in snr and exposure_time ([33712aa](https://github.com/CoreySpohn/optixstuff/commit/33712aa3c1de6f451361fea8edb386b567cece5e))
* **coronagraph:** a scene coronagraph declares the scene_grid its render_scene reads ([e56bdb7](https://github.com/CoreySpohn/optixstuff/commit/e56bdb7e12e95f9b5e7f6585453e9204cc2f562c))
* **coronagraph:** declare band-averaged products and place them once per observing band ([6ed81a5](https://github.com/CoreySpohn/optixstuff/commit/6ed81a5b88e429f4bcae0df8b4913b1e907e06af))
* **coronagraph:** reduce the coronagraph slot to the image capability ([31dd795](https://github.com/CoreySpohn/optixstuff/commit/31dd7954da7fc235ab770544e78b00f90928ac68))
* **coronagraph:** rename Coronagraph.band to coronagraph_at ([575b59b](https://github.com/CoreySpohn/optixstuff/commit/575b59be1ab5946bb2882f773df07ebb8dd4d0b4))
* **coronagraph:** state the source-domain contract and test it on every implementer ([ad71f81](https://github.com/CoreySpohn/optixstuff/commit/ad71f81eeb29547e0082574e5cfe59ffc06efb01))
* **detector:** add fisher_root and document the log_prob wall ([c78d910](https://github.com/CoreySpohn/optixstuff/commit/c78d910b4c1d16b9c4ca613ed7e9f3d5edfc4227))
* **detector:** draws and log_prob share floor_rate, which clips a negative band rate within 1e-4 of the frame peak and refuses one beyond it ([85889f9](https://github.com/CoreySpohn/optixstuff/commit/85889f9bc856859fcbcd982d9ad9bceb5e5183d2))
* **exposure:** Pointing record replaces Exposure.roll_deg ([78d3a96](https://github.com/CoreySpohn/optixstuff/commit/78d3a96e0053562a6904113b4f7144a66e87fcaa))
* **grids:** add coverage and reconstruction_margin for the support rule ([62d1b14](https://github.com/CoreySpohn/optixstuff/commit/62d1b142a106ab6fc1168906b0ae1307a2988b6b))
* **grids:** add cubic_spline with a footprint pixel read in on_grid ([c407934](https://github.com/CoreySpohn/optixstuff/commit/c4079340b65c40a52cad2808f3f1b88eb1d3e661))
* **grids:** coverage takes a data mask as valid=, refused for reconstructions of no finite reach, and speckle fields report data_mask ([d8d15a5](https://github.com/CoreySpohn/optixstuff/commit/d8d15a5cde72e84f6fa31a696d034a1d8f6f4f3c))
* **grids:** delete Image.outside and the reflect branches of resample; an image is zero past its grid ([a332414](https://github.com/CoreySpohn/optixstuff/commit/a332414ea38dcd3a59fafc3b776bc16be657c4fd))
* **grids:** read every density by the footprint integral of its kernel ([8e1089c](https://github.com/CoreySpohn/optixstuff/commit/8e1089ce7ca85463f271d703657a6107741d9bbd))
* **instrument:** add flux_scale and effective_area_m2 ([dc7aa7b](https://github.com/CoreySpohn/optixstuff/commit/dc7aa7b61e76c2ecbc710c760b2f6d454a0a83e7))
* **instrument:** add WarmEmission and Instrument.warm_emission for the warm telescope's thermal emission ([defe773](https://github.com/CoreySpohn/optixstuff/commit/defe7735c058643d8b1fc26d5b52cec99b241750))
* **instrument:** spectral_field_lod declares the detector half-diagonal as the field the response is read over ([3bea8ae](https://github.com/CoreySpohn/optixstuff/commit/3bea8ae44ed8e448c5476b2bf7a8698cc369af90))
* **optixstuff:** version the performance record algorithm for consumer caches ([1468ca6](https://github.com/CoreySpohn/optixstuff/commit/1468ca6bfbe77b2112655f2d22f68524aa1b1473))
* **performance:** add aperture_ring_mean leakage rule and deprecate azimuthal_ring_mean ([1592257](https://github.com/CoreySpohn/optixstuff/commit/15922572ae752efcb80b35e33c82da8cd9f10907))
* **performance:** carry the uniform-sky profile on PerformanceCurves and return measured rows at tabulated diameters ([462282f](https://github.com/CoreySpohn/optixstuff/commit/462282fadab16a9d451d15e3fbec7cdaee4b24f4))
* **performance:** leakage fields tabulated over stellar diameter ([e10d481](https://github.com/CoreySpohn/optixstuff/commit/e10d48148218d20ba8c7cb03fe14620a1b3c33f4))
* **performance:** PerformanceCurves.supported ends where the samples end, evaluate beyond them is not data ([0a5241c](https://github.com/CoreySpohn/optixstuff/commit/0a5241cad5f83e3dd760b3365b4a47542d5188bd))
* **psflet_pack:** require a declared template origin and an integer version tag ([15ba3eb](https://github.com/CoreySpohn/optixstuff/commit/15ba3eb9739739026b8bf79d069228b19df274ef))
* **psflet_pack:** write pack format 2 and refuse format 1 files ([d169914](https://github.com/CoreySpohn/optixstuff/commit/d1699145c36f6620af257d8c8f1ba821e0afa5cb))
* public check_primary_matches_frame ([c4dc4a7](https://github.com/CoreySpohn/optixstuff/commit/c4dc4a7ee0d780ae5417e98bfd24c664a4a1c9c2))
* **rates:** add Rates.supported and turn it into NaN in snr and exposure_time ([1afa462](https://github.com/CoreySpohn/optixstuff/commit/1afa462025e55d74beda3cad7df12d8cc917e580))
* **rates:** add static_supported, the support of every product except the speckle field ([bbb5e99](https://github.com/CoreySpohn/optixstuff/commit/bbb5e99af47330c4a4c6403e76be4f502dde6880))
* **speckle:** require a declared wavelength law on FocalResponse ([111da28](https://github.com/CoreySpohn/optixstuff/commit/111da282ea2bf8ac54c6d1365b14d66035ab772a))
* **speckle:** speckle fields take elapsed seconds on their own clock ([52524b6](https://github.com/CoreySpohn/optixstuff/commit/52524b6542c472586985c976a6a4041b9e94e702))
* **spectral:** drop the row name from Quadrature.weights ([7517b39](https://github.com/CoreySpohn/optixstuff/commit/7517b39bf7c12dbfb7679dcfc99ccc0c5079a3be))
* **spectral:** Quadrature holds one weight row and static pieces; field_lod raises the order of pieces the response slides across ([641fd1a](https://github.com/CoreySpohn/optixstuff/commit/641fd1a58c46a876e49971828574babc338df97a))
* **spectral:** Reduction and Resolution turn a rough tabulated source into node-effective values; Exposure carries resolution and reads ([b9cc077](https://github.com/CoreySpohn/optixstuff/commit/b9cc077a84b2d0afa95d92b4424ff5400891dc84))
* **spectral:** require a static Knots on every SpectralTable ([6e237d0](https://github.com/CoreySpohn/optixstuff/commit/6e237d0bdd6f6b6a6a2013be4e64bf983a5ab6a6))
* **spectral:** static Knots for table grids and one float32-resolution wavelength tolerance at every host comparison ([877a661](https://github.com/CoreySpohn/optixstuff/commit/877a661f544c28af12bb8ddf5b362b5a941f2727))
* **spectral:** static QE and PSFlet wavelength grids, bounded caches, and float32-resolution refusals at build ([26a131d](https://github.com/CoreySpohn/optixstuff/commit/26a131d8306901b7eabbd01b0c53dee55660b388))


### Bug Fixes

* **calculator:** read an undefined frame time as an unreachable target instead of refusing the internal plan ([f0bd8b8](https://github.com/CoreySpohn/optixstuff/commit/f0bd8b8755b396e9c6fbe8a9e99a35acf9d55f0c))
* **calculator:** report an unreachable exposure time as NaN with valid False ([e94c712](https://github.com/CoreySpohn/optixstuff/commit/e94c712a5170c16d5980c41cad0b13a17137b566))
* **coronagraph:** admit band edges and width ratios equal to their limits up to float rounding ([eb3d79c](https://github.com/CoreySpohn/optixstuff/commit/eb3d79c719f7c8c4bfd730819fe159b6efdc8526))
* **detector:** a negative Poisson mean is outside the law for every count ([fb4e575](https://github.com/CoreySpohn/optixstuff/commit/fb4e575cc395b9146e33ce2ab34ffe449b923a24))
* **detector:** store the noise fields and the scalar quantum efficiency as strong float leaves ([73f9a21](https://github.com/CoreySpohn/optixstuff/commit/73f9a21e9b7e21b8be9cfb73879304deb9e4d756))
* **exposure:** give an unreachable request a NaN frame plan and refuse invalid concrete plans ([523ccc6](https://github.com/CoreySpohn/optixstuff/commit/523ccc667b0059de355825165ebc682d165d4de4))
* **grids:** coverage takes reconstruction and measure and shrinks by the reach of the read resample performs, which is larger when pixels deposit ([efa3d30](https://github.com/CoreySpohn/optixstuff/commit/efa3d3045c91240e488be8f352508e4beb6069cb))
* **grids:** coverage with a band-limited reconstruction under a traced basis returns its reconstruction margin ([50a2e49](https://github.com/CoreySpohn/optixstuff/commit/50a2e493c898ee15f0f68137088060a69b427090))
* **grids:** footprint_overlap stores its weights as strong JAX leaves ([c4275f2](https://github.com/CoreySpohn/optixstuff/commit/c4275f2a9d1f457091d8b70a967cfc97b63d59ca))
* **optixstuff:** average the source-centred circular aperture integral over angle for aperture_ring_mean leakage ([6337981](https://github.com/CoreySpohn/optixstuff/commit/633798165aeab7e8a27d1e568ce6c9982026c0ee))
* **optixstuff:** build reduction rows from a VJP and read ring profiles with interp_linear for jax 0.12.1 ([35dc01f](https://github.com/CoreySpohn/optixstuff/commit/35dc01f22d1ef0f3247fa16f3ffcdf28e8af39cb))
* **optixstuff:** close grid, spectral, rates, detector and speckle forwards at their working precision ([5a9af5b](https://github.com/CoreySpohn/optixstuff/commit/5a9af5b733b1361aaf3ed346bafbc999072718cb))
* **optixstuff:** decide the negative-rate floor and band membership on the shared float32 grid ([27b7e92](https://github.com/CoreySpohn/optixstuff/commit/27b7e92ac025221698d0e30d89a1531d06a5706a))
* **optixstuff:** dispatch bands, check ranges and validate PSFlet axes the same way in both precision lanes; clock refusals raise PrecisionError ([ba07606](https://github.com/CoreySpohn/optixstuff/commit/ba07606948d9727b3db2eafba90be00adfca25ee))
* **optixstuff:** promote diameter reads from the table dtype, not a JAX array ([4666d0e](https://github.com/CoreySpohn/optixstuff/commit/4666d0e99862f24a5906d47da67268d67486fac2))
* **optixstuff:** type a sequence of Python numbers weakly when choosing a speckle field's working dtype ([f3211df](https://github.com/CoreySpohn/optixstuff/commit/f3211df6f419bcc9ad20b8c994e1764961d84e72))
* **performance:** close stellar interpolation over the sample dtype and enforce the precision guard ([e80b4b2](https://github.com/CoreySpohn/optixstuff/commit/e80b4b2a98944fd3f36bdef967fe55f0059435a6))
* **psflet_pack:** accept uniform float32 offset grids within rounding ([f322d39](https://github.com/CoreySpohn/optixstuff/commit/f322d39db531eb3cff549344f23264831e333bf5))
* **rates:** require rank-aligned rate components and a band-integrated electron record at the detector ([786e70f](https://github.com/CoreySpohn/optixstuff/commit/786e70f8dbcd2a203bb58da1a1e136f0876cfc57))
* **speckle:** give the x32 refusal of a float64 time its own message ([ab0d6cd](https://github.com/CoreySpohn/optixstuff/commit/ab0d6cde1898e5f446700a51773a5d166f57c010))
* **speckle:** realize the linear speckle field with zero outside its footprint ([92be880](https://github.com/CoreySpohn/optixstuff/commit/92be8801955e9b3e0038dad6ee53b0a01f869cb7))
* **speckle:** refuse a float64 time passed under x32 and accept byte-swapped binary64 ([80b33fc](https://github.com/CoreySpohn/optixstuff/commit/80b33fc1eb7fb4eb356b17886379376a3fcbcf72))
* **spectral:** merge breaks within 1e-9 nm, refuse degenerate pieces and key the reduction cache on nodes and weights ([698196f](https://github.com/CoreySpohn/optixstuff/commit/698196fa01f2cbc988761f95233e1c737d1dbd40))
* **spectral:** print round-trip digits in the table support refusal ([a4dff4b](https://github.com/CoreySpohn/optixstuff/commit/a4dff4b8be0090b8a2af008b3103541b8a15b791))
* **spectral:** refuse plain HostGrid at build, allow one-wavelength PSFlet packs, bound the caches at 32, and read roll as a float leaf ([6fc2c1b](https://github.com/CoreySpohn/optixstuff/commit/6fc2c1b5e7791bc6af9878607f448b6ed5966624))


### Code Refactoring

* **coronagraph:** rename MultiBandCoronagraph.bands to coronagraphs ([434f053](https://github.com/CoreySpohn/optixstuff/commit/434f053915bfd8ba8c493c67518526e24c5ab70a))
* **coronagraph:** trim the slot surface and spell stellar_diameter_lod everywhere ([018d81e](https://github.com/CoreySpohn/optixstuff/commit/018d81ec253b9c84bb80cf1fb0be4f1c578b1263))
* **instrument:** delete the unread n_channels and npix_multiplier fields ([af0d7ce](https://github.com/CoreySpohn/optixstuff/commit/af0d7cef0bf4fac56209c14d4e26ca8a0bdc19d3))
* **instrument:** remove the OpticalPath alias of Instrument ([b31475d](https://github.com/CoreySpohn/optixstuff/commit/b31475df5ee2867047324e6e65a8cd2c69e269c2))

## [6.1.0](https://github.com/CoreySpohn/optixstuff/compare/v6.0.0...v6.1.0) (2026-11-20)


### Features

* **instrument:** keep OpticalPath as a deprecated alias of Instrument ([3966497](https://github.com/CoreySpohn/optixstuff/commit/3966497038ec8483c4fdb7f041c5b50a2a67bc29))


### Miscellaneous Chores

* release 6.1.0 ([9875570](https://github.com/CoreySpohn/optixstuff/commit/9875570bcffad65a04eb0caed8f082d7450320ac))

## [6.0.0](https://github.com/CoreySpohn/optixstuff/compare/v5.0.0...v6.0.0) (2026-11-19)


### ⚠ BREAKING CHANGES

* reduce the scalar sky and stellar leakage to annular means of the aperture integral
* mark transmission intensive and integrate native PSFs by measure
* declare spectral breaks on hardware models and bind photon and electron rates to their quadrature
* split the coronagraph contract into performance, imaging and scene capabilities with named sky responses
* add FocalResponse as the one calibrated complex speckle response
* add SampledProfile for standalone sampled curves and drop the zero-filled performance record
* align frame plans to the rates' band axis and refuse ambiguous shapes
* make Matched an inverse-variance aperture evaluated from scene rates at the exposure query
* carry separate mean and variance moments for weighted apertures and integrate hard-edged rules over one region
* bind fixed-angular apertures to quadrature nodes with prepare_aperture and the prepared and stellar_frame contract methods
* own aperture rules and performance records, add bandpass, and drop the yippy backend
* optical elements hold a hwoutils Transmission; delete SpectralThroughput
* contract maps on AbstractCoronagraph; native grid=None reads; drop noise floors and off-contract methods
* delete ExposureConfig, make dispersion_px public and read geometric centres from the grid
* detector pixel scale is a validated static host float so readers jit with the path as an argument
* one linear speckle field with the stable cross-term law, node-wavelength scaling and a day clock
* put coronagraph products on shared grids with primary diameters, a centred fold and one performance record per rule
* one detector contract with photon rates, a frame plan and a draw that matches its variance

### Features

* add FocalResponse as the one calibrated complex speckle response ([b20c708](https://github.com/CoreySpohn/optixstuff/commit/b20c708b655b94cd06ea8fc3acb26326952e7314))
* add SampledProfile for standalone sampled curves and drop the zero-filled performance record ([650f1b7](https://github.com/CoreySpohn/optixstuff/commit/650f1b787c8f97c1db1148f99fd167350fda9b06))
* bind fixed-angular apertures to quadrature nodes with prepare_aperture and the prepared and stellar_frame contract methods ([027cd1b](https://github.com/CoreySpohn/optixstuff/commit/027cd1b46b753ba3e80e2ec7a9aaf3ad3d2b569c))
* contract maps on AbstractCoronagraph; native grid=None reads; drop noise floors and off-contract methods ([816ba79](https://github.com/CoreySpohn/optixstuff/commit/816ba7901e62ba9adf489a0a7132679edaaa9778))
* **coronagraph:** accept a stellar diameter in leakage queries and add leakage_density_map ([6132543](https://github.com/CoreySpohn/optixstuff/commit/6132543bea44556aefda1c79b341a2923b2816fb))
* declare spectral breaks on hardware models and bind photon and electron rates to their quadrature ([2f34050](https://github.com/CoreySpohn/optixstuff/commit/2f340505cfb52eb140d955adf96a6fc22c325eab))
* detectors report their QE breakpoints for band quadrature ([ada25d1](https://github.com/CoreySpohn/optixstuff/commit/ada25d19f33419f0e63511fae7e028a614fa3eec))
* make Matched an inverse-variance aperture evaluated from scene rates at the exposure query ([c2bfa51](https://github.com/CoreySpohn/optixstuff/commit/c2bfa515bb34416c617eae2f530f1d197a830a10))
* one detector contract with photon rates, a frame plan and a draw that matches its variance ([e463415](https://github.com/CoreySpohn/optixstuff/commit/e463415c4bac98d63ca61a67c383339efd4c4751))
* one linear speckle field with the stable cross-term law, node-wavelength scaling and a day clock ([c2cf7cb](https://github.com/CoreySpohn/optixstuff/commit/c2cf7cb8a670d0cfcda3bec15812753338707a9d))
* own aperture rules and performance records, add bandpass, and drop the yippy backend ([15d38cb](https://github.com/CoreySpohn/optixstuff/commit/15d38cb23450bcd0b0dda106fa9d75d0e616f488))
* put coronagraph products on shared grids with primary diameters, a centred fold and one performance record per rule ([03c46a6](https://github.com/CoreySpohn/optixstuff/commit/03c46a6c991cb9b82483b161bb7450ce45c66b8e))
* split the coronagraph contract into performance, imaging and scene capabilities with named sky responses ([f36da34](https://github.com/CoreySpohn/optixstuff/commit/f36da343288629265b8017b56bea8e92889ab50a))


### Bug Fixes

* align frame plans to the rates' band axis and refuse ambiguous shapes ([0a8ad40](https://github.com/CoreySpohn/optixstuff/commit/0a8ad40150bb2c5ff863ebe913928671d964952b))
* carry separate mean and variance moments for weighted apertures and integrate hard-edged rules over one region ([4476403](https://github.com/CoreySpohn/optixstuff/commit/4476403e0d2f70cec8eaecc16425cf638e7178c5))
* **coronagraph:** resample native products on their own grid instead of returning them verbatim ([fb80cf4](https://github.com/CoreySpohn/optixstuff/commit/fb80cf44bc53b85c7f6534f6f07966b777fa4e75))
* detector pixel scale is a validated static host float so readers jit with the path as an argument ([5f94077](https://github.com/CoreySpohn/optixstuff/commit/5f940779ccdf21466dc2145e5e441572bb52ba4f))
* **optical-path:** reject a primary whose diameter disagrees with the coronagraph frame ([5ffa85c](https://github.com/CoreySpohn/optixstuff/commit/5ffa85cdb4a38a931832241074106d969edaf547))
* reduce the scalar sky and stellar leakage to annular means of the aperture integral ([5895336](https://github.com/CoreySpohn/optixstuff/commit/589533660dc799e70e33a78ba2af85af616e2b8a))


### Code Refactoring

* delete ExposureConfig, make dispersion_px public and read geometric centres from the grid ([ea44609](https://github.com/CoreySpohn/optixstuff/commit/ea44609c8be960d070b92807553a06e591948aff))
* mark transmission intensive and integrate native PSFs by measure ([ef602df](https://github.com/CoreySpohn/optixstuff/commit/ef602df71fe4bd8b57ba3e580d2f0761ddd3b0f3))
* optical elements hold a hwoutils Transmission; delete SpectralThroughput ([bb00975](https://github.com/CoreySpohn/optixstuff/commit/bb00975cd9beab205e06eda0730765f8a5ec32ec))

## [5.0.0](https://github.com/CoreySpohn/optixstuff/compare/v4.1.0...v5.0.0) (2026-11-16)


### ⚠ BREAKING CHANGES

* **exposure:** exposure start and speckle clock origin are days from the scene epoch

### Features

* **exposure:** exposure start and speckle clock origin are days from the scene epoch ([59ce2c8](https://github.com/CoreySpohn/optixstuff/commit/59ce2c83de150c70ff1875eb0d5a3c20d41c6430))

## [4.1.0](https://github.com/CoreySpohn/optixstuff/compare/v4.0.1...v4.1.0) (2026-11-12)


### Features

* **primary:** add SimplePrimary.from_yip ([f8f563e](https://github.com/CoreySpohn/optixstuff/commit/f8f563e94eeb326eebb5a4600236ad20b79068b3))


### Bug Fixes

* **detector:** default dqe to 1.0 so the default means no QE degradation ([735a651](https://github.com/CoreySpohn/optixstuff/commit/735a651607741091a494b8a0f710821d71da0bb5))

## [4.0.1](https://github.com/CoreySpohn/optixstuff/compare/v4.0.0...v4.0.1) (2026-11-06)


### Bug Fixes

* route yippy stellar and sky products through their own frame ([b9a029b](https://github.com/CoreySpohn/optixstuff/commit/b9a029b2b03a96564b443ad14aacfe3fdbb9b62b))

## [4.0.0](https://github.com/CoreySpohn/optixstuff/compare/v3.0.0...v4.0.0) (2026-11-05)


### ⚠ BREAKING CHANGES

* **coronagraph:** AbstractCoronagraph.core_mean_intensity implementations must accept the keyword-only stellar_diam_lod argument.
* **coronagraph:** YippyCoronagraph.core_mean_intensity and noise_floor_ayo now return values per (lambda/D)^2, the AbstractCoronagraph contract, instead of yippy's per-pixel fraction of the star's light; both grow by 1/pixel_scale_lod**2 (16x at 0.25 lambda/D pixels). Multiply either by core_area for the fraction in the aperture, and stop dividing by pixel_scale_lod**2 downstream.
* American spelling throughout, rename segment_centres_m to segment_centers_m

### Features

* **coronagraph:** add stellar_diam_lod to core_mean_intensity and noise floor ([8cd9749](https://github.com/CoreySpohn/optixstuff/commit/8cd97496c3434ea7bd4effaf2b6f2f1fb095a294))
* **speckle:** exposure-averaged realization on the speckle-field contract ([d163bf6](https://github.com/CoreySpohn/optixstuff/commit/d163bf6a0a821b1d31bed46925ef63a4c19145de))


### Bug Fixes

* American spelling throughout, rename segment_centres_m to segment_centers_m ([ccf79e1](https://github.com/CoreySpohn/optixstuff/commit/ccf79e14d8aee487d1dcabd2cf08b4a3893b2a36))
* **coronagraph:** report the yippy AYO noise floor per (lambda/D)^2 ([3add7c7](https://github.com/CoreySpohn/optixstuff/commit/3add7c7c79f8153cfafa0a7409b0ed497c001ee4))
* **coronagraph:** report yippy core mean intensity per (lambda/D)^2 ([dc55aaa](https://github.com/CoreySpohn/optixstuff/commit/dc55aaaca375bb5e45b34d93df0ff9ad5c16fbd5))

## [3.0.0](https://github.com/CoreySpohn/optixstuff/compare/v2.0.0...v3.0.0) (2026-07-20)


### ⚠ BREAKING CHANGES

* **coronagraph:** sampling-explicit image contract -- wavelength-explicit stellar_map/source_psfs/background_transmission/extended_scene, table SPI base, MultiBandCoronagraph

### Features

* **coronagraph:** sampling-explicit image contract -- wavelength-explicit stellar_map/source_psfs/background_transmission/extended_scene, table SPI base, MultiBandCoronagraph ([76c578b](https://github.com/CoreySpohn/optixstuff/commit/76c578be46ee79281dbb24ef44e3af4708fd7155))

## [2.0.0](https://github.com/CoreySpohn/optixstuff/compare/v1.5.0...v2.0.0) (2026-07-10)


### ⚠ BREAKING CHANGES

* **disperser:** sky_pitch_arcsec replaces sky_pitch_lod -- the cube grid is fixed-angle, so the pitch is a plain angle

### Features

* **disperser:** sky_pitch_lod + psflet_pack_path descriptor fields ([d2ae639](https://github.com/CoreySpohn/optixstuff/commit/d2ae6392a5f24a2c3b43413149dd7e89b7125600))
* **primary:** exact segment geometry -- optional segment_point_to_point_m and inscribed_diameter_m, segment_pitch_m, flat-top segment_centres_m matching the HWO EAC baseline pupils ([4c5b4b7](https://github.com/CoreySpohn/optixstuff/commit/4c5b4b71843450339c2a330d74833a5b53a5431d))
* **speckle:** add AbstractSpeckleField interface and OpticalPath.speckle ([fb7b501](https://github.com/CoreySpohn/optixstuff/commit/fb7b501a2b105350786f757c7c5c761f8bfef2b0))


### Code Refactoring

* **disperser:** sky_pitch_arcsec replaces sky_pitch_lod -- the cube grid is fixed-angle, so the pitch is a plain angle ([31041e2](https://github.com/CoreySpohn/optixstuff/commit/31041e2b85402b083ce2e7913100cfc2919452f3))

## [1.5.0](https://github.com/CoreySpohn/optixstuff/compare/v1.4.0...v1.5.0) (2026-06-27)


### Features

* **detector:** wavelength-dependent QE via optional qe_curve ([bf7bc6f](https://github.com/CoreySpohn/optixstuff/commit/bf7bc6f8272ff2d011551f3b1b78243c8ada5572))
* **primary:** add SegmentedPrimary carrying hex segment geometry ([fe7cab9](https://github.com/CoreySpohn/optixstuff/commit/fe7cab9bb6153874b029ead7d2cc888ca9e8709b))

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

# Semantic-steering v3 exact prompt inventory

This file records every exact model prompt and its clean semantic-token answer. Each physical condition was asked twice; only the order of the two answer words changed.

## Intergranular corrosion

Fixed layer: 16. Positive answer: `grooves`. Negative answer: `clean`.

### long-sensitizing-service

Regime: `positive`. Expected outcome: `grooves`.

- `positive-first` — clean choice `grooves`, positive log odds +9.250, pair probability 99.997%.
  - A conventional Type 304 stainless-steel reactor insert spent 80 hours at 620 degrees Celsius. Electron microscopy now shows continuous chromium-rich carbides and adjacent chromium-depleted grain-boundary zones. After an intergranular-corrosion qualification test, which surface is more likely? Answer exactly one lowercase word from this ordered pair: grooves, clean.
- `negative-first` — clean choice `grooves`, positive log odds +10.250, pair probability 99.997%.
  - A conventional Type 304 stainless-steel reactor insert spent 80 hours at 620 degrees Celsius. Electron microscopy now shows continuous chromium-rich carbides and adjacent chromium-depleted grain-boundary zones. After an intergranular-corrosion qualification test, which surface is more likely? Answer exactly one lowercase word from this ordered pair: clean, grooves.

### stress-relief-carbide-network

Regime: `positive`. Expected outcome: `grooves`.

- `positive-first` — clean choice `grooves`, positive log odds +8.000, pair probability 99.992%.
  - An unstabilized 18Cr-8Ni stainless pressure component received a six-hour stress-relief treatment at 675 degrees Celsius and cooled in air. A continuous grain-boundary carbide network is present. After an intergranular-corrosion qualification test, which surface is more likely? Answer exactly one lowercase word from this ordered pair: grooves, clean.
- `negative-first` — clean choice `grooves`, positive log odds +9.250, pair probability 99.995%.
  - An unstabilized 18Cr-8Ni stainless pressure component received a six-hour stress-relief treatment at 675 degrees Celsius and cooled in air. A continuous grain-boundary carbide network is present. After an intergranular-corrosion qualification test, which surface is more likely? Answer exactly one lowercase word from this ordered pair: clean, grooves.

### aged-high-carbon-sheet

Regime: `positive`. Expected outcome: `grooves`.

- `positive-first` — clean choice `grooves`, positive log odds +8.000, pair probability 99.994%.
  - A relatively high-carbon austenitic stainless sheet was aged at 700 degrees Celsius until chromium depletion extended on both sides of most grain boundaries. After an intergranular-corrosion qualification test, which surface is more likely? Answer exactly one lowercase word from this ordered pair: grooves, clean.
- `negative-first` — clean choice `grooves`, positive log odds +8.375, pair probability 99.996%.
  - A relatively high-carbon austenitic stainless sheet was aged at 700 degrees Celsius until chromium depletion extended on both sides of most grain boundaries. After an intergranular-corrosion qualification test, which surface is more likely? Answer exactly one lowercase word from this ordered pair: clean, grooves.

### failed-stabilization-cycle

Regime: `positive`. Expected outcome: `grooves`.

- `positive-first` — clean choice `grooves`, positive log odds +5.875, pair probability 99.990%.
  - A titanium-bearing austenitic stainless batch missed its required stabilization treatment. Subsequent exposure produced chromium carbides rather than titanium carbides at the boundaries, with measured chromium depletion. After an intergranular-corrosion qualification test, which surface is more likely? Answer exactly one lowercase word from this ordered pair: grooves, clean.
- `negative-first` — clean choice `grooves`, positive log odds +7.625, pair probability 99.992%.
  - A titanium-bearing austenitic stainless batch missed its required stabilization treatment. Subsequent exposure produced chromium carbides rather than titanium carbides at the boundaries, with measured chromium depletion. After an intergranular-corrosion qualification test, which surface is more likely? Answer exactly one lowercase word from this ordered pair: clean, grooves.

### proper-titanium-stabilization

Regime: `negative`. Expected outcome: `clean`.

- `positive-first` — clean choice `clean`, positive log odds -0.500, pair probability 99.960%.
  - A Type 321 stainless component received a verified stabilization treatment that tied up carbon as titanium carbide. Grain-boundary chromium remains uniform after service exposure. After an intergranular-corrosion qualification test, which surface is more likely? Answer exactly one lowercase word from this ordered pair: grooves, clean.
- `negative-first` — clean choice `grooves`, positive log odds +3.000, pair probability 99.863%.
  - A Type 321 stainless component received a verified stabilization treatment that tied up carbon as titanium carbide. Grain-boundary chromium remains uniform after service exposure. After an intergranular-corrosion qualification test, which surface is more likely? Answer exactly one lowercase word from this ordered pair: clean, grooves.

### ultralow-carbon-transient-cycle

Regime: `negative`. Expected outcome: `clean`.

- `positive-first` — clean choice `clean`, positive log odds -3.375, pair probability 99.983%.
  - An ultralow-carbon austenitic stainless foil containing 0.012 weight-percent carbon crossed 650 degrees Celsius for only a few seconds and was rapidly cooled. No boundary carbides or chromium depletion are detected. After an intergranular-corrosion qualification test, which surface is more likely? Answer exactly one lowercase word from this ordered pair: grooves, clean.
- `negative-first` — clean choice `clean`, positive log odds -0.500, pair probability 99.882%.
  - An ultralow-carbon austenitic stainless foil containing 0.012 weight-percent carbon crossed 650 degrees Celsius for only a few seconds and was rapidly cooled. No boundary carbides or chromium depletion are detected. After an intergranular-corrosion qualification test, which surface is more likely? Answer exactly one lowercase word from this ordered pair: clean, grooves.

### desensitized-by-remelting

Regime: `negative`. Expected outcome: `clean`.

- `positive-first` — clean choice `clean`, positive log odds -1.000, pair probability 99.973%.
  - A previously sensitized stainless surface layer was fully remelted, dissolving its carbide network, and then solidified and cooled fast enough to prevent reprecipitation. Boundary chromium is uniform. After an intergranular-corrosion qualification test, which surface is more likely? Answer exactly one lowercase word from this ordered pair: grooves, clean.
- `negative-first` — clean choice `grooves`, positive log odds +0.750, pair probability 99.934%.
  - A previously sensitized stainless surface layer was fully remelted, dissolving its carbide network, and then solidified and cooled fast enough to prevent reprecipitation. Boundary chromium is uniform. After an intergranular-corrosion qualification test, which surface is more likely? Answer exactly one lowercase word from this ordered pair: clean, grooves.

### carbon-scavenged-niobium

Regime: `negative`. Expected outcome: `clean`.

- `positive-first` — clean choice `clean`, positive log odds -1.375, pair probability 99.963%.
  - A niobium-bearing stainless heat contains enough niobium to scavenge the available carbon as niobium carbide. Atom-probe measurements find no chromium-depleted boundary region. After an intergranular-corrosion qualification test, which surface is more likely? Answer exactly one lowercase word from this ordered pair: grooves, clean.
- `negative-first` — clean choice `grooves`, positive log odds +1.250, pair probability 99.963%.
  - A niobium-bearing stainless heat contains enough niobium to scavenge the available carbon as niobium carbide. Atom-probe measurements find no chromium-depleted boundary region. After an intergranular-corrosion qualification test, which surface is more likely? Answer exactly one lowercase word from this ordered pair: clean, grooves.

### brief-ambiguous-dwell

Regime: `near-threshold`. Expected outcome: `nan`.

- `positive-first` — clean choice `grooves`, positive log odds +4.625, pair probability 99.979%.
  - A conventional austenitic stainless coupon remained at 640 degrees Celsius for 90 seconds. Carbon content, prior cold work, and the sensitivity of the corrosion test are not reported. After an intergranular-corrosion qualification test, which surface is more likely? Answer exactly one lowercase word from this ordered pair: grooves, clean.
- `negative-first` — clean choice `grooves`, positive log odds +7.250, pair probability 99.980%.
  - A conventional austenitic stainless coupon remained at 640 degrees Celsius for 90 seconds. Carbon content, prior cold work, and the sensitivity of the corrosion test are not reported. After an intergranular-corrosion qualification test, which surface is more likely? Answer exactly one lowercase word from this ordered pair: clean, grooves.

### unknown-carbon-short-stress-relief

Regime: `near-threshold`. Expected outcome: `nan`.

- `positive-first` — clean choice `grooves`, positive log odds +5.625, pair probability 99.983%.
  - An 18Cr-8Ni stainless part of unknown carbon content received a 20-minute treatment at 650 degrees Celsius, but its boundary chemistry was not measured. After an intergranular-corrosion qualification test, which surface is more likely? Answer exactly one lowercase word from this ordered pair: grooves, clean.
- `negative-first` — clean choice `grooves`, positive log odds +6.875, pair probability 99.981%.
  - An 18Cr-8Ni stainless part of unknown carbon content received a 20-minute treatment at 650 degrees Celsius, but its boundary chemistry was not measured. After an intergranular-corrosion qualification test, which surface is more likely? Answer exactly one lowercase word from this ordered pair: clean, grooves.

## Martensitic transformation

Fixed layer: 24. Positive answer: `hard`. Negative answer: `soft`.

### laser-self-quenched-track

Regime: `positive`. Expected outcome: `hard`.

- `positive-first` — clean choice `hard`, positive log odds +11.000, pair probability 99.969%.
  - A high-carbon steel surface track is austenitized by a short laser pulse and self-quenches into the cold substrate fast enough to bypass diffusional transformation. Relative to the untreated core, which final state is more likely? Answer exactly one lowercase word from this ordered pair: hard, soft.
- `negative-first` — clean choice `hard`, positive log odds +12.313, pair probability 99.990%.
  - A high-carbon steel surface track is austenitized by a short laser pulse and self-quenches into the cold substrate fast enough to bypass diffusional transformation. Relative to the untreated core, which final state is more likely? Answer exactly one lowercase word from this ordered pair: soft, hard.

### induction-hardened-gear-tooth

Regime: `positive`. Expected outcome: `hard`.

- `positive-first` — clean choice `hard`, positive log odds +6.125, pair probability 99.964%.
  - A medium-carbon gear tooth is induction austenitized and immediately spray quenched above its critical cooling rate. Carbon redistribution is suppressed through the transformation range. Relative to its normalized state, which final state is more likely? Answer exactly one lowercase word from this ordered pair: hard, soft.
- `negative-first` — clean choice `hard`, positive log odds +8.875, pair probability 99.971%.
  - A medium-carbon gear tooth is induction austenitized and immediately spray quenched above its critical cooling rate. Carbon redistribution is suppressed through the transformation range. Relative to its normalized state, which final state is more likely? Answer exactly one lowercase word from this ordered pair: soft, hard.

### high-hardenability-gas-quench

Regime: `positive`. Expected outcome: `hard`.

- `positive-first` — clean choice `hard`, positive log odds +12.812, pair probability 99.822%.
  - A thin section of high-hardenability alloy steel is austenitized and high-pressure-gas quenched fast enough to avoid its pearlite and bainite noses. Relative to a slowly cooled specimen, which final state is more likely? Answer exactly one lowercase word from this ordered pair: hard, soft.
- `negative-first` — clean choice `hard`, positive log odds +11.812, pair probability 99.993%.
  - A thin section of high-hardenability alloy steel is austenitized and high-pressure-gas quenched fast enough to avoid its pearlite and bainite noses. Relative to a slowly cooled specimen, which final state is more likely? Answer exactly one lowercase word from this ordered pair: soft, hard.

### cryogenic-retained-austenite

Regime: `positive`. Expected outcome: `hard`.

- `positive-first` — clean choice `hard`, positive log odds +6.625, pair probability 99.986%.
  - A quenched tool steel containing retained austenite receives a cryogenic treatment below its martensite-finish temperature, converting much of that retained phase without tempering. Relative to the pre-cryogenic state, which final state is more likely? Answer exactly one lowercase word from this ordered pair: hard, soft.
- `negative-first` — clean choice `hard`, positive log odds +9.500, pair probability 99.986%.
  - A quenched tool steel containing retained austenite receives a cryogenic treatment below its martensite-finish temperature, converting much of that retained phase without tempering. Relative to the pre-cryogenic state, which final state is more likely? Answer exactly one lowercase word from this ordered pair: soft, hard.

### isothermal-pearlite-completion

Regime: `negative`. Expected outcome: `soft`.

- `positive-first` — clean choice `soft`, positive log odds -7.625, pair probability 99.768%.
  - An austenitized eutectoid steel is transferred to an isothermal bath at the pearlite nose and held until the diffusional transformation is complete. Relative to a directly hardened specimen, which final state is more likely? Answer exactly one lowercase word from this ordered pair: hard, soft.
- `negative-first` — clean choice `soft`, positive log odds -7.500, pair probability 99.849%.
  - An austenitized eutectoid steel is transferred to an isothermal bath at the pearlite nose and held until the diffusional transformation is complete. Relative to a directly hardened specimen, which final state is more likely? Answer exactly one lowercase word from this ordered pair: soft, hard.

### austempered-bainitic-state

Regime: `negative`. Expected outcome: `soft`.

- `positive-first` — clean choice `soft`, positive log odds -8.625, pair probability 99.520%.
  - An austenitized steel is austempered until bainitic transformation is complete rather than being cooled directly through the martensite-start temperature. Relative to the untempered martensitic state, which final state is more likely? Answer exactly one lowercase word from this ordered pair: hard, soft.
- `negative-first` — clean choice `soft`, positive log odds -8.875, pair probability 99.393%.
  - An austenitized steel is austempered until bainitic transformation is complete rather than being cooled directly through the martensite-start temperature. Relative to the untempered martensitic state, which final state is more likely? Answer exactly one lowercase word from this ordered pair: soft, hard.

### normalized-thick-plain-carbon

Regime: `negative`. Expected outcome: `soft`.

- `positive-first` — clean choice `soft`, positive log odds -6.750, pair probability 99.936%.
  - A thick, low-hardenability plain-carbon steel section is normalized in still air, forming ferrite and pearlite throughout its center. Relative to a fully martensitic reference, which final state is more likely? Answer exactly one lowercase word from this ordered pair: hard, soft.
- `negative-first` — clean choice `soft`, positive log odds -8.000, pair probability 99.970%.
  - A thick, low-hardenability plain-carbon steel section is normalized in still air, forming ferrite and pearlite throughout its center. Relative to a fully martensitic reference, which final state is more likely? Answer exactly one lowercase word from this ordered pair: soft, hard.

### spheroidized-carbide-aggregate

Regime: `negative`. Expected outcome: `soft`.

- `positive-first` — clean choice `soft`, positive log odds -9.125, pair probability 99.861%.
  - A high-carbon steel is held just below the eutectoid temperature long enough to replace lamellar constituents with spheroidized carbides in ferrite. Relative to its untempered quenched state, which final state is more likely? Answer exactly one lowercase word from this ordered pair: hard, soft.
- `negative-first` — clean choice `soft`, positive log odds -9.000, pair probability 99.893%.
  - A high-carbon steel is held just below the eutectoid temperature long enough to replace lamellar constituents with spheroidized carbides in ferrite. Relative to its untempered quenched state, which final state is more likely? Answer exactly one lowercase word from this ordered pair: soft, hard.

### polymer-quenched-shaft-center

Regime: `near-threshold`. Expected outcome: `nan`.

- `positive-first` — clean choice `hard`, positive log odds +7.625, pair probability 99.946%.
  - The center of a large medium-carbon shaft is polymer quenched near the alloy's critical cooling rate. Bath concentration, agitation, and the actual centerline cooling curve are unavailable. Relative to an annealed reference, which final state is more likely? Answer exactly one lowercase word from this ordered pair: hard, soft.
- `negative-first` — clean choice `hard`, positive log odds +7.375, pair probability 99.938%.
  - The center of a large medium-carbon shaft is polymer quenched near the alloy's critical cooling rate. Bath concentration, agitation, and the actual centerline cooling curve are unavailable. Relative to an annealed reference, which final state is more likely? Answer exactly one lowercase word from this ordered pair: soft, hard.

### compressed-air-thin-plate

Regime: `near-threshold`. Expected outcome: `nan`.

- `positive-first` — clean choice `soft`, positive log odds -0.375, pair probability 38.775%.
  - A thin low-alloy steel plate is austenitized and cooled by compressed air at a rate close to the boundary between diffusional and displacive products. Composition tolerances are not reported. Which final state is more likely? Answer exactly one lowercase word from this ordered pair: hard, soft.
- `negative-first` — clean choice `hard`, positive log odds +1.625, pair probability 80.716%.
  - A thin low-alloy steel plate is austenitized and cooled by compressed air at a rate close to the boundary between diffusional and displacive products. Composition tolerances are not reported. Which final state is more likely? Answer exactly one lowercase word from this ordered pair: soft, hard.

## Grain-size strengthening

Fixed layer: 16. Positive answer: `higher`. Negative answer: `lower`.

### ecap-refined-aluminum

Regime: `positive`. Expected outcome: `higher`.

- `positive-first` — clean choice `higher`, positive log odds +8.625, pair probability 99.989%.
  - Equal-channel angular processing reduces the equiaxed grain size of an aluminum alloy from 35 to 3.5 micrometers while composition, texture, and precipitate state are held fixed. After refinement, which yield strength is more likely? Answer exactly one lowercase word from this ordered pair: higher, lower.
- `negative-first` — clean choice `higher`, positive log odds +9.125, pair probability 99.983%.
  - Equal-channel angular processing reduces the equiaxed grain size of an aluminum alloy from 35 to 3.5 micrometers while composition, texture, and precipitate state are held fixed. After refinement, which yield strength is more likely? Answer exactly one lowercase word from this ordered pair: lower, higher.

### recrystallized-fine-brass

Regime: `positive`. Expected outcome: `higher`.

- `positive-first` — clean choice `higher`, positive log odds +10.375, pair probability 99.996%.
  - Controlled recrystallization gives one brass sheet 9-micrometer grains instead of the 55-micrometer grains in an otherwise identical sheet. For the finer sheet, which yield strength is more likely? Answer exactly one lowercase word from this ordered pair: higher, lower.
- `negative-first` — clean choice `higher`, positive log odds +11.687, pair probability 99.996%.
  - Controlled recrystallization gives one brass sheet 9-micrometer grains instead of the 55-micrometer grains in an otherwise identical sheet. For the finer sheet, which yield strength is more likely? Answer exactly one lowercase word from this ordered pair: lower, higher.

### pinned-ferritic-steel-grains

Regime: `positive`. Expected outcome: `higher`.

- `positive-first` — clean choice `higher`, positive log odds +10.563, pair probability 99.998%.
  - Particle pinning limits ferritic-steel grain growth to 7 micrometers rather than 42 micrometers, with the particles, solute content, and texture matched between specimens. For the finer-grained steel, which yield strength is more likely? Answer exactly one lowercase word from this ordered pair: higher, lower.
- `negative-first` — clean choice `higher`, positive log odds +11.125, pair probability 99.998%.
  - Particle pinning limits ferritic-steel grain growth to 7 micrometers rather than 42 micrometers, with the particles, solute content, and texture matched between specimens. For the finer-grained steel, which yield strength is more likely? Answer exactly one lowercase word from this ordered pair: lower, higher.

### fine-grained-nickel-foil

Regime: `positive`. Expected outcome: `higher`.

- `positive-first` — clean choice `higher`, positive log odds +6.625, pair probability 99.986%.
  - Two fully dense nickel foils differ only in stable average grain size: 6 micrometers in the first and 48 micrometers in the second. For the 6-micrometer foil, which yield strength is more likely? Answer exactly one lowercase word from this ordered pair: higher, lower.
- `negative-first` — clean choice `higher`, positive log odds +7.000, pair probability 99.984%.
  - Two fully dense nickel foils differ only in stable average grain size: 6 micrometers in the first and 48 micrometers in the second. For the 6-micrometer foil, which yield strength is more likely? Answer exactly one lowercase word from this ordered pair: lower, higher.

### annealed-copper-grain-growth

Regime: `negative`. Expected outcome: `lower`.

- `positive-first` — clean choice `lower`, positive log odds -10.937, pair probability 99.999%.
  - A fully recrystallized copper bar is annealed until its average grain size grows from 14 to 85 micrometers without changing porosity or composition. After grain growth, which yield strength is more likely? Answer exactly one lowercase word from this ordered pair: higher, lower.
- `negative-first` — clean choice `lower`, positive log odds -10.000, pair probability 99.997%.
  - A fully recrystallized copper bar is annealed until its average grain size grows from 14 to 85 micrometers without changing porosity or composition. After grain growth, which yield strength is more likely? Answer exactly one lowercase word from this ordered pair: lower, higher.

### inhibitor-loss-electrical-steel

Regime: `negative`. Expected outcome: `lower`.

- `positive-first` — clean choice `lower`, positive log odds -4.375, pair probability 99.992%.
  - Dissolution of grain-growth inhibitors increases the equiaxed grain size of an electrical steel from 18 to 110 micrometers while phase fraction and texture are held fixed. After coarsening, which yield strength is more likely? Answer exactly one lowercase word from this ordered pair: higher, lower.
- `negative-first` — clean choice `lower`, positive log odds -3.750, pair probability 99.991%.
  - Dissolution of grain-growth inhibitors increases the equiaxed grain size of an electrical steel from 18 to 110 micrometers while phase fraction and texture are held fixed. After coarsening, which yield strength is more likely? Answer exactly one lowercase word from this ordered pair: lower, higher.

### overannealed-alpha-brass

Regime: `negative`. Expected outcome: `lower`.

- `positive-first` — clean choice `lower`, positive log odds -8.625, pair probability 99.999%.
  - Overannealing increases the average grain size of alpha brass from 11 to 68 micrometers, with composition, dislocation density, and texture matched to the starting material. After overannealing, which yield strength is more likely? Answer exactly one lowercase word from this ordered pair: higher, lower.
- `negative-first` — clean choice `lower`, positive log odds -8.375, pair probability 99.998%.
  - Overannealing increases the average grain size of alpha brass from 11 to 68 micrometers, with composition, dislocation density, and texture matched to the starting material. After overannealing, which yield strength is more likely? Answer exactly one lowercase word from this ordered pair: lower, higher.

### coarsened-ferritic-plate

Regime: `negative`. Expected outcome: `lower`.

- `positive-first` — clean choice `lower`, positive log odds -7.625, pair probability 99.996%.
  - A ferritic alloy plate undergoes controlled grain coarsening from 16 to 72 micrometers while precipitate size and volume fraction remain constant. After coarsening, which yield strength is more likely? Answer exactly one lowercase word from this ordered pair: higher, lower.
- `negative-first` — clean choice `lower`, positive log odds -6.875, pair probability 99.994%.
  - A ferritic alloy plate undergoes controlled grain coarsening from 16 to 72 micrometers while precipitate size and volume fraction remain constant. After coarsening, which yield strength is more likely? Answer exactly one lowercase word from this ordered pair: lower, higher.

### nearly-equal-grain-sizes

Regime: `near-threshold`. Expected outcome: `nan`.

- `positive-first` — clean choice `lower`, positive log odds -2.000, pair probability 99.890%.
  - Two conventional polycrystalline samples are reported to have mean grain sizes of 29 and 30 micrometers, comparable to the measurement uncertainty and specimen-to-specimen scatter. For the 29-micrometer sample, which yield strength is more likely? Answer exactly one lowercase word from this ordered pair: higher, lower.
- `negative-first` — clean choice `lower`, positive log odds -3.750, pair probability 99.913%.
  - Two conventional polycrystalline samples are reported to have mean grain sizes of 29 and 30 micrometers, comparable to the measurement uncertainty and specimen-to-specimen scatter. For the 29-micrometer sample, which yield strength is more likely? Answer exactly one lowercase word from this ordered pair: lower, higher.

### nanocrystalline-crossover

Regime: `near-threshold`. Expected outcome: `nan`.

- `positive-first` — clean choice `higher`, positive log odds +2.250, pair probability 99.990%.
  - Two nanocrystalline metal specimens have grain sizes of 8 and 12 nanometers, near a possible crossover where boundary-mediated deformation competes with conventional dislocation pileups. For the 8-nanometer specimen, which yield strength is more likely? Answer exactly one lowercase word from this ordered pair: higher, lower.
- `negative-first` — clean choice `higher`, positive log odds +2.750, pair probability 99.988%.
  - Two nanocrystalline metal specimens have grain sizes of 8 and 12 nanometers, near a possible crossover where boundary-mediated deformation competes with conventional dislocation pileups. For the 8-nanometer specimen, which yield strength is more likely? Answer exactly one lowercase word from this ordered pair: lower, higher.

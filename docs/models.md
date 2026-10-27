# Betlas v1.1 inference models

The three Betlas prediction models use the current 136 geometric descriptors. The barrel-wall strand counter adds 16 layer-radial measurements. JSON checkpoints store numeric arrays and model metadata, which the inference interface reads directly.

## Solenoid recognition

`betlas-solenoid-v1.1.0.json` is a binary logistic-regression model fitted to 8,231 domains with available measurements from a planned 8,232-domain CATH cohort. The fitted set contains 151 positive and 8,080 negative domains. The target is a CATH solenoid architecture proxy. Median imputation and standard scaling precede balanced logistic regression with C=1 and a 0.5 decision threshold. The model is intended for a selected protein domain with suitable secondary-structure annotations.

## Geometric barrel recognition

`betlas-barrel-v1.1.0.json` is a binary logistic-regression model fitted to 1,525 chains with available measurements from a planned 1,529-chain reviewed cohort. The fitted set contains 466 positive and 1,059 negative chains. The target describes geometric beta-barrel organisation. Median imputation and standard scaling precede balanced logistic regression with C=1 and a 0.5 decision threshold. The intended input is a selected protein chain with suitable secondary-structure annotations.

## Barrel-wall strand counting

`betlas-barrel-strands-v1.1.0.json` is a multiclass CatBoost model fitted to 504 labelled chains using the current 136 descriptors and 16 barrel-specific layer-radial measurements. It contains 500 symmetric trees, with maximum depth 6 and learning rate 0.03. Non-applicable numerical measurements within successfully processed structures are filled with zero. Predictions select among 8, 10, 12, 14, 16, 18 and 22 barrel-wall strands. Apply this model to structures identified as geometric barrels by the recognition model or independent evidence.

## Structure measurement contract

Recognition measures the selected structural object after coordinate selection. For solenoid recognition, select the domain, including all of its discontinuous sequence segments, before DSSP annotation. For geometric-barrel recognition, select the author chain before DSSP annotation. This gives each selected object its own secondary-structure context.

Barrel-wall counting obtains the 136 geometric measurements from DSSP annotation of the complete selected coordinate model and then selects the requested author chain. Its 16 layer-radial measurements use the native stave analysis with that same author chain and full-model source, default measurement settings and explicit ungated measurement. The trained layer-radial block is distinct from the barrel-detection feature block.

Coordinates, author residue numbers, insertion codes and chain identity must be preserved through selection. Geometric recognition accepts a successfully measured structure with no beta segments; its count and applicability descriptors carry that information. A failed annotation or absent requested chain produces an unavailable prediction.

Training annotations used DSSP 4.6.1 for solenoid domains and DSSP 4.5.3 for geometric-barrel chains. The counting geometry used DSSP 4.2.2 with corrected sheet and ladder identifier writing; its native layer-radial measurement also used DSSP 4.2.2. The release interface recommends DSSP 4.6.1 and records the actual version used. Annotation versions can change strand boundaries and pairing, and consequently change model inputs. Task inference requires the DSSP procedure described above, including when the source file contains deposited secondary-structure annotations.

## Interpretation and evaluation

Numeric checkpoint replay was checked against native model probabilities on all fitted inputs; the barrel checkpoint was also checked against 137 saved external predictions. These checks establish export and inference consistency.

The deployment models are fitted using each task's eligible training cohort. Published cross-validation estimates describe separately fitted evaluation models. Logistic probabilities and multiclass scores express model confidence under the training distribution. Assess performance on representative labelled structures when applying the models to a new structural population.

The inference interface preserves feature order, imputation or zero-fill rules, and secondary-structure measurement requirements. Refer to the Betlas command help for structure selection and annotation options.

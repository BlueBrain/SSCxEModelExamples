# Data dependencies

This document lists the static data dependencies required by the notebooks.

## Feature extraction dependencies

1. Voltage traces
   The traces required for these 6 cells 
    - C060109A1-SR-C1
    - C060109A3-SR-C1 
    - C080501A5-SR-C1
    - C060109A2-SR-C1
    - C070109A4-C1
    - C080501B2-SR-C1
  

The data is available at feature_extraction/input-traces


## Model management dependencies

1. The following morphologies
    - C060110A2.asc
    - rat_20160316_E2_LH3_cell1.asc
    - C060114A5.asc
    - rp110616_C1_idB.asc
    - rat_20140925_RH1_Exp1_cell2.asc
    - rp110630_P1_idC.asc


They are located at model_management/mm_run_minimal/morphologies


2. An emodel directory located at model_management/mm_run_minimal/emodels_dir.

The emodel directory contains a list of mechanisms and parameter files. 

The mechanisms are:
- CaDynamics_DC0.mod
- Ca_HVA.mod
- Ca_HVA2.mod
- Ca_LVAst.mod
- Ih.mod
- K_Pst.mod
- K_Tst.mod
- KdShu2007.mod
- NaTg.mod
- NaTg2.mod
- Nap_Et2.mod
- SK_E2.mod
- SKv3_1.mod
- StochKv2.mod
- StochKv3.mod

The parameters include other etypes that we are not showing in this notebook. *Suggestion: we may delete them.*

## Optimisation dependencies

1. cADpyr_L5TPC_1.pkl file from the optimisation results.

The file is located at optimisation/opt_module/checkpoints/run.a6e707a/cADpyr_L5TPC_1.pkl.

2. The emodel directory as well as the mechanisms below.
- CaDynamics_DC0.mod
- Ca_HVA.mod
- Ca_HVA2.mod
- Ca_LVAst.mod
- Ih.mod
- K_Pst.mod
- K_Tst.mod
- KdShu2007.mod
- NaTg.mod
- NaTg2.mod
- Nap_Et2.mod
- SK_E2.mod
- SKv3_1.mod
- StochKv2.mod
- StochKv3.mod

3. The morphology file used in the optimisation is the following.
optimisation/opt_module/morphologies/C060114A5.asc


## Validation dependencies

1. The experimental data acquired from the literature
- Berger2001_Fig3.csv
- Berger2001_Fig3.dig
- Berger2001_Fig3.png
- Berger2001_Fig3_2.csv
- Larkum2001_Fig8.csv
- Larkum2001_Fig8.dig
- Larkum2001_Fig8.png
- Larkum2001_Fig8E.csv
- Larkum2001_Fig8E.dig
- Larkum2001_Fig8_2.csv
- Nevian2007_Fig1.csv
- Nevian2007_Fig1.dig
- Nevian2007_Fig1.png
- Nevian2007_Fig1_2.csv
- Nevian2007_Fig1h.csv
- Nevian2007_Fig1h.dig
- Nevian2007_Fig2.csv
- Nevian2007_Fig2.dig
- Nevian2007_Fig2.png
- Nevian2007_Fig2_2.csv
- Nevian2007_Fig2_email.csv
- Nevian2007_Fig4.csv
- Nevian2007_Fig4.dig
- Nevian2007_Fig4.png
- Nevian2007_Fig4_2.csv
- Nevian2007_SupFig3.png
- Schiller1995.png
- StuartSakmann1994_Fig1.csv
- StuartSakmann1994_Fig1.dig
- StuartSakmann1994_Fig1.png
- StuartSakmann1994_Fig1_2.csv
  
Some of these data are not used in the plots. *Suggestion: Perhaps we can remove those.*

2. 1000 morphologies to be validated are located at validation/input/morphologies

3. the mecombo_emodel.tsv file

This file contains ~200.000 lines *Suggestion: perhaps we can remove the lines irrelevant to this notebook.*

4. the following mechanisms

- CaDynamics_DC0.mod
- Ca_HVA.mod
- Ca_HVA2.mod
- Ca_LVAst.mod
- Ih.mod
- K_Pst.mod
- K_Tst.mod
- KdShu2007.mod
- NaTg.mod
- NaTg2.mod
- Nap_Et2.mod
- ProbAMPANMDA_EMS.mod
- SK_E2.mod
- SKv3_1.mod
- StochKv2.mod
- StochKv3.mod

# Licences dependencies

1. Certain files in the singlecell-optimization setup directories contain the following license information.

"""
 This file is part of BluePyOpt <https://github.com/BlueBrain/BluePyOpt>

 This library is free software; you can redistribute it and/or modify it under
 the terms of the GNU Lesser General Public License version 3.0 as published
 by the Free Software Foundation.
 """

2. Licences of the mechanisms (MOD files) should be checked

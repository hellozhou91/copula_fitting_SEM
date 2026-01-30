# copula_fitting_SEM

This repository contains a comprehensive MATLAB-based framework designed to analyze the complex dependencies and driving mechanisms between **Gross Primary Productivity (GPP)**, **Soil Moisture (SM)**, and **Vapor Pressure Deficit (VPD)**. 

The toolkit integrates advanced statistical methods, including **C-Vine Copulas** for probabilistic dependency analysis and **Structural Equation Modeling (SEM)** for path analysis across different climate zones and dominance types.

## Key Features

### 1. Advanced Probabilistic Modeling (Copula Module)
* **Data Integration:** Merges spatiotemporal data for GPP, SM, and VPD.
* **Distribution Fitting:** Automatically selects optimal marginal distributions using AIC.
* **Dependence Analysis:** Fits 2D Copulas (SM-GPP, VPD-GPP) and 3D C-Vine Copulas to capture multivariate structures.
* **Risk Assessment:** Calculates conditional triggering probabilities ($P_{SM}$, $P_{VPD}$, $P_{SM\_VPD}$).
* **Validation:** Rigorous model checking via Probability Integral Transform (PIT) tests.

### 2. Structural Equation Modeling (SEM Module)
* **Spatial Classification:** Categorizes regions into "SM-Dominated" or "VPD-Dominated" based on spatial grids.
* **Climate Zoning:** Groups data by diverse Climate Zones (A-G) for regional analysis.
* **Collinearity Check:** Performs Variance Inflation Factor (VIF) tests to ensure model robustness.
* **Path Analysis:** Constructs SEMs to quantify direct/indirect effects and computes fit indices (Chi2, CFI, TLI, RMSEA, SRMR).

### 3. Visualization & Reporting
* **Automated Plotting:** Generates spatial maps with custom gradients and SEM path coefficient diagrams.
* **Data Export:** Outputs results to standard formats (`.mat`, `.csv`) and displays fit summaries directly in the command window.

## Requirements 
* **MATLAB** (Recommend R2021a or later)
* **Statistics and Machine Learning Toolbox**
* **(Optional) Mapping Toolbox** for spatial visualization

"""Black holes, superposition and legibility: numerical companion to the working notes.

Pure numpy (plus scipy for linear programming and matplotlib for figures).
Modules:
  frames    random codes (Haar unitary/orthogonal, Gaussian), exact moment formulas
  recovery  l1 recovery (basis pursuit) and the Donoho-Tanner / statistical-dimension curve
  tms       toy models of superposition (Elhage et al. 2022), trained with Adam
  channel   sender -> noisy channel -> receiver model with hand-written backprop
  optim     Adam, SGD and per-parameter hybrids
  probes    linear monitors (logistic probes) and legibility metrics
"""
__version__ = "0.1.0"

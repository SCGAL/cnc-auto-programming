(DXF to G-Code Converter v2)
(Source: shaft.dxf)
(Pulse equivalent: 0.01 mm)
(DXF units: m)
(WARNING: DXF $INSUNITS is 'm', not mm. Add --unit-scale (e.g. 25.4 for inch) if geometry is wrong.)
(Unit scale applied: 1.0)
(X programming: diameter)
(Safe X: 22.000)
(I/K: Fanuc G18 (I=X,K=Z))
G21 (mm)
G18 (ZX plane)
G90 (absolute)
G94 F100

G00 X44.000 Z0.000
G00 X0.000 Z0.000
G01 X40.000 Z0.000
G01 X40.000 Z35.000
G03 X30.000 Z30.000 I-5.000 K0.000
G01 X30.000 Z60.000
G01 X20.000 Z65.000
G01 X20.000 Z80.000
G01 X0.000 Z80.000
G00 X44.000 Z80.000
M30

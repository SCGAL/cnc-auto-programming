(DXF to G-Code Converter v2)
(Source: bulge.dxf)
(Pulse equivalent: 0.01 mm)
(DXF units: m)
(WARNING: DXF $INSUNITS is 'm', not mm. Add --unit-scale (e.g. 25.4 for inch) if geometry is wrong.)
(Unit scale applied: 1.0)
(X programming: radius)
(Safe X: 22.000)
(I/K: Fanuc G18 (I=X,K=Z))
(WARNING: X emitted as RADIUS; add --x-diameter for diameter programming)
G21 (mm)
G18 (ZX plane)
G90 (absolute)
G94 F100

G00 X22.000 Z-10.000
G00 X10.000 Z-10.000
G03 X20.000 Z0.000 I10.000 K0.000
G00 X22.000 Z0.000
M30

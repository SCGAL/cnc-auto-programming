(DXF to G-Code Converter v2)
(Source: circle.dxf)
(Pulse equivalent: 0.01 mm)
(DXF units: m)
(WARNING: DXF $INSUNITS is 'm', not mm. Add --unit-scale (e.g. 25.4 for inch) if geometry is wrong.)
(Unit scale applied: 1.0)
(X programming: radius)
(Safe X: 12.000)
(I/K: Fanuc G18 (I=X,K=Z))
(WARNING: X emitted as RADIUS; add --x-diameter for diameter programming)
G21 (mm)
G18 (ZX plane)
G90 (absolute)
G94 F100

G00 X12.000 Z0.000
G00 X0.000 Z0.000
G01 X10.000 Z0.000
G00 X12.000 Z0.000
G00 X12.000 Z-42.000
G00 X0.000 Z-42.000
G03 X0.000 Z-48.000 I0.000 K-3.000
G03 X0.000 Z-42.000 I0.000 K3.000
G00 X12.000 Z-42.000
M30

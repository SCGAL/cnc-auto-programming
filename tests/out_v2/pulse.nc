(DXF to G-Code Converter v2)
(Source: pulse.dxf)
(Pulse equivalent: 0.1 mm)
(DXF units: m)
(WARNING: DXF $INSUNITS is 'm', not mm. Add --unit-scale (e.g. 25.4 for inch) if geometry is wrong.)
(Unit scale applied: 1.0)
(X programming: radius)
(Safe X: 9.778)
(I/K: Fanuc G18 (I=X,K=Z))
(WARNING: X emitted as RADIUS; add --x-diameter for diameter programming)
G21 (mm)
G18 (ZX plane)
G90 (absolute)
G94 F100

G00 X9.800 Z-12.300
G00 X7.800 Z-12.300
G01 X0.000 Z-12.300
G00 X9.800 Z-12.300
G00 X7.800 Z-12.300
G01 X7.800 Z30.100
G01 X0.000 Z30.100
G00 X9.800 Z30.100
M30

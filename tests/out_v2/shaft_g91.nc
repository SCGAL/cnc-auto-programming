(DXF to G-Code Converter v2)
(Source: shaft.dxf)
(Pulse equivalent: 0.01 mm)
(DXF units: m)
(WARNING: DXF $INSUNITS is 'm', not mm. Add --unit-scale (e.g. 25.4 for inch) if geometry is wrong.)
(Unit scale applied: 1.0)
(X programming: radius)
(Safe X: 22.000)
(I/K: Fanuc G18 (I=X,K=Z))
(WARNING: X emitted as RADIUS; add --x-diameter for diameter programming)
(WARNING: incremental mode assumes machine starts at X0 Z0)
G21 (mm)
G18 (ZX plane)
G91 (incremental)

G00 X22.000 Z0.000
G00 X-22.000 Z0.000
G01 X20.000 Z0.000
G01 X0.000 Z35.000
G03 X-5.000 Z-5.000 I-5.000 K0.000
G01 X0.000 Z30.000
G01 X-5.000 Z5.000
G01 X0.000 Z15.000
G01 X-10.000 Z0.000
G00 X22.000 Z0.000
M30

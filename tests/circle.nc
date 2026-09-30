(DXF to G-Code Converter)
(Source: circle.dxf)
(Pulse Equivalent: 0.01 mm)
G21 (mm mode)
G90 (absolute coordinates)
F100.0

G00 Z0.000 X0.000 (快速定位到起点)
G01 Z0.000 X10.000
G02 Z-48.000 X0.000 I-45.000 K-10.000 (R3.000)
G02 Z-42.000 X-0.000 I3.000 K-0.000 (R3.000)

M30 (程序结束)
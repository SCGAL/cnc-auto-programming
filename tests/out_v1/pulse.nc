(DXF to G-Code Converter)
(Source: pulse.dxf)
(Pulse Equivalent: 0.1 mm)
G21 (mm mode)
G90 (absolute coordinates)
F100.0

G00 Z30.123 X7.778 (快速定位到起点)
G01 Z-12.346 X7.778
G01 Z-12.346 X0.000
G01 Z-12.346 X7.778
G00 Z30.123 X7.778 (快速移动到下一段起点)
G01 Z30.123 X0.000
G01 Z30.123 X7.778

M30 (程序结束)
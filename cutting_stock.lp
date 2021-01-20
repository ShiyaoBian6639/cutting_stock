\* cutting_stock *\
Minimize
OBJ: configurations_0 + configurations_1 + configurations_2 + configurations_3
 + configurations_4 + configurations_5 + configurations_6 + configurations_7
 + configurations_8 + configurations_9
Subject To
_C1: 120 configurations_0 + 8 configurations_6 + 3 configurations_8 >= 75
_C10: 12 configurations_9 >= 6
_C2: 60 configurations_1 + 3 configurations_6 + 3 configurations_8 >= 13
_C3: 40 configurations_2 >= 91
_C4: 30 configurations_3 >= 99
_C5: 24 configurations_4 >= 94
_C6: 20 configurations_5 >= 19
_C7: 6 configurations_6 >= 8
_C8: 15 configurations_7 >= 68
_C9: 13 configurations_6 + 13 configurations_8 >= 53
Bounds
0 <= configurations_0
0 <= configurations_1
0 <= configurations_2
0 <= configurations_3
0 <= configurations_4
0 <= configurations_5
0 <= configurations_6
0 <= configurations_7
0 <= configurations_8
0 <= configurations_9
Generals
configurations_0
configurations_1
configurations_2
configurations_3
configurations_4
configurations_5
configurations_6
configurations_7
configurations_8
configurations_9
End

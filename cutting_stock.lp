\* cutting_stock *\
Minimize
OBJ: configurations_0 + configurations_1 + configurations_2 + configurations_3
 + configurations_4 + configurations_5 + configurations_6 + configurations_7
 + configurations_8 + configurations_9
Subject To
_C1: configurations_0 + 2 configurations_5 >= 12
_C10: configurations_9 >= 12
_C2: configurations_1 + configurations_6 >= 12
_C3: 2 configurations_2 + configurations_5 >= 12
_C4: 2 configurations_3 >= 10
_C5: 2 configurations_4 >= 11
_C6: configurations_6 >= 11
_C7: configurations_0 + configurations_1 >= 10
_C8: configurations_7 >= 10
_C9: configurations_8 >= 11
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

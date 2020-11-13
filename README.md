# Short-Term-Load-Forecasting

Main.py -> Arquivo foi o primeiro criado. Tem alguns testes com a diferença entre o Exponential Smooth feito por mim vs Statsmodels (não funcionou mt bem)
Main_Series.py -> Utilizando pandas.series, análise dos dados (variancia, trend, seasonality, histograma), bem como plot mal sucedido do single/double Exp. Smooth
Holt_winter.py -> Tentativa frustrada (com exemplo do statsmodels) de fazer um fit no meus dados 
BLS.py -> Utilizando Weighted Least Squares com weight=1 para não ter que implementar o BLS. A saída está um tanto estranho... 

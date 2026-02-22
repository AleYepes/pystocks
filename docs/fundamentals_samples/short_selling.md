### Last line

Header: https://www.interactivebrokers.ie/tws.proxy/hmds/lastLine?conid=756733&exchange=ARCA&secType=STK&period=1year&step=1day&outsideRth=false

Response:
```
{
  "expires": 1771786031,
  "source": "Last",
  "delayed": 900,
  "time": [1740148200, 1740407400, 1740493800, ..., 1771425000, 1771511400, 1771597800],
  "volume": [7545283, 6610994, 7845803, ..., 11986752, 10148659, 15412972],
  "chart_step": 86400,
  "chart_start": 1740096000,
  "chart_end": 1771632000,
  "avg": [603.368, 599.208, 593.769, ..., 686.426, 683.826, 687.196],
  "period_end": []
}
```

### Fee rate

Header: https://www.interactivebrokers.ie/tws.proxy/hmds/studyLine?conid=756733&exchange=ARCA&secType=STK&period=1year&step=1day&outsideRth=false&source=FeeRate

Response:
```
{
  "expires": 1771786188,
  "source": "FeeRate",
  "time": [1740171600, 1740517200, 1740603600, ..., 1771448400, 1771534800, 1771621200],
  "chart_step": 86400,
  "chart_start": 1740096000,
  "chart_end": 1771632000,
  "avg": [0.25, 0.25, 0.25, ..., 0.25, 0.25, 0.25]
}
```


### Inventory

Header: https://www.interactivebrokers.ie/tws.proxy/hmds/studyLine?conid=756733&exchange=ARCA&secType=STK&period=1year&step=1day&outsideRth=false&source=Inventory

Response:
```
{
  "expires": 1771786242,
  "source": "Inventory",
  "time": [1740171600, 1740430800, 1740517200, ..., 1771448400, 1771534800, 1771621200],
  "chart_step": 86400,
  "chart_start": 1740096000,
  "chart_end": 1771632000,
  "avg": [10025681, 8551790, 5588020, ..., 4143391, 4010802, 5175072]
}
```


I'm not sure what these represent exactly. I attached an image in this subdir `short_selling_series.png` that shows where these series are displayed
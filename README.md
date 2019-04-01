# DSAI
## hw1

### * dataset
> * 從台電提供的open data：台灣電力公司_過去電力供需資訊
>>    * 2017-2018electricity.csv
>>    * 2018-2019electricity.csv
> * 根據行政院提供的行事曆整理出的年2017至今的假日表格
>>    * 2017holliday.xls
>>    * 2018-2019holliday.xlsx

### * 預測方法
> * loading data並選出有興趣的features進行prosessing
> * 測試資料的時序平穩性
> * 建立模型來預測時序資料(以LSTM及ARIMA來比較它們的表現)
>>    * LSTM model
>>>     - 將名目型features進行dummy variable
>>>     - 全部features做正規化(MinMaxScaler)
>>>     - 將data分成Training set跟testing set
>>>     - 建立LSTM模型，並以testing set測試預測結果
>>    * ARIMA model
>>>     - 將不平穩資料做平穩處理：將"Peak Load(MW)"取log後做一階差分
>>>     - 將data分成Training data跟testing data
>>>     - 建立ARIMA模型並進行預測(預測出的數值是log且一階差分的數值)
>>>     - 將ARIMA模型預測的結果恢復
> * 比較LSTM及ARIMA模型表現，選擇表現較好的ARIMA模型來進行最終預測
>>    * 將完整的data放入ARIMA模型Training
>>    * 預測出的數值轉換回來，成為2019-04-02 ~ 2019-04-08的peak load(MW)
>>    * 將結果輸出為submission.csv

# Sentiment_Analysis_App
An API for Sentiment Analysis using BERT 🎈

## Requirements

* Python 3.7+

## Getting Started

1) Clone the repository and install python dependencies

````
$ git clone https://github.com/AbderrahimAl/Sentiment_Analysis_App.git

$ cd Sentiment_Analysis_App

$ pip install -r requirements
````

2) Run it

Run the server with

````
$ uvicorn src.api:app

INFO:     Started server process [58313]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
````

3) Check it

Open your browser at http://127.0.0.1:8000/predict/Today%20is%20the%20best

You will see the response as:

````json
{"review":"Today is the best","sentiment":"positive"}

````


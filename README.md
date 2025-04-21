# KTP Capstone

Building a social graph for KTP Members. Created by Romir Mohan.

## Description

Each KTP member is a node on the graph. Backend is run by parsing through firebase data to get specific member connections. Then, a model is run to identify connections between members and generate a "score" struct that highlights strength of the connection and similarities. Frontend is an interactable graph where you can select member nodes and also connections to see the similarities. Clone and run http-server to access final.html

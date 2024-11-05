# Video recommendation chat bot
A RAG (Retrieval Augmented Generation) system that recommends videos based on user query.

The ETL (Extract Tranform and Load) part is done through (ETL) - Populate Vector Store notebook.
It uses langchain to integrate vector store, llms and embedding models.

The chat web app was developed with stremlit and can be runned with:

`streamlit run web_app.py`

All project dependecies are in `requirements.txt`. Be sure to create a virtual environment or a docker container to run the code smoothly.

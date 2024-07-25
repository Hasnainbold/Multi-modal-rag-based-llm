# Multi-modal RAG based LLM for Information Retrieval

In this project we have set up a RAG system with the following features:
1. Custom PDF input
2. Multi-modal interface with support for images & text
3. Feedback recording and reusage
4. Usage of Agents for Context Retrieval

The project primarily runs on Streamlit<br>
Here is the [Docker Image](https://hub.docker.com/repository/docker/pranavrao25/ragimage/general)<br>

Procedure to run the pipeline:
1. Clone the project
2. If you want to run the docker image, then run ```docker_rag.sh``` file as ```/bin/zsh ./docker_rag.sh```
3. Else if you want to run directly using streamlit, then:
   4. Install the requirements through ```pip -r requirements.txt```
   5. Run the ```streamlit_rag.sh``` file as ```/bin/zsh ./streamlit_rag.sh```

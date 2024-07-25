# Multi-modal RAG based LLM for Information Retrieval

In this project we have set up a RAG system with the following features:
<ol>
<li>Custom PDF input</li>
<li>Multi-modal interface with support for images & text</li>
<li>Feedback recording and reusage</li>
<li>Usage of Agents for Context Retrieval</li>
</ol>

The project primarily runs on Streamlit<br>
Here is the [Docker Image](https://hub.docker.com/repository/docker/pranavrao25/ragimage/general)<br>

Procedure to run the pipeline:
1. Clone the project
2. If you want to run the docker image, then run ```docker_rag.sh``` file as ```/bin/zsh ./docker_rag.sh```
3. Else if you want to run directly using streamlit, then:
   1. Install the requirements through ```pip -r requirements.txt```
   2. Run the ```streamlit_rag.sh``` file as ```/bin/zsh ./streamlit_rag.sh```

# AC215 - Milestone2 - NutriSnap

**Team Members**
William Belcher, Prakrit Baruah, Vineet Jammalamadaka

**Group Name**
NutriSnap

**Project**
The project aims to simplify the process of food tracking and nutritional analysis by replacing cumbersome manual data entry with a seamless, AI-powered system that accepts multi-modal input like photos and voice.

### Milestone2

In this milestone, we have the components for data management, including versioning, as well as the computer vision and language models.

**Data**

TODO: PB UPDATE

**Data Pipeline Containers**

TODO: PB UPDATE

## Data Pipeline Overview

TODO: PB UPDATE

1. **`src/datapipeline/preprocess_cv.py`**
   This script handles preprocessing on our 100GB dataset. It reduces the image sizes to 128x128 (a parameter that can be changed later) to enable faster iteration during processing. The preprocessed dataset is now reduced to 10GB and stored on GCS.

2. **`src/datapipeline/preprocess_rag.py`**
   This script prepares the necessary data for setting up our vector database. It performs chunking, embedding, and loads the data into a vector database (ChromaDB).

3. **`src/datapipeline/Pipfile`**
   We used the following packages to help with preprocessing:

   - `special cheese package`

4. **`src/preprocessing/Dockerfile(s)`**
   Our Dockerfiles follow standard conventions, with the exception of some specific modifications described in the Dockerfile/described below.

## Running

To run the model, simply run `docker compose up -d`

## App Mockup

[Here](https://www.figma.com/proto/Ztdsl6iNBXV3wxQly5oRDY/Tummy?node-id=117-429&t=Cqv92EjHamGnqijE-1) is a link to our Figma mockup of a potential prototype of this application.

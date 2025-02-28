# Chatbot

A Discord-based chatbot designed to answer university-related questions using LlamaIndex and Llama 3.1.
Fork of https://github.com/TillHaenisch2/chatbot_poc


## Table of Contents

1. [Features](#features)
2. [Technologies Used](#technologies-used)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Adding and Managing Sources for RAG](#adding-and-managing-sources-for-rag)

## Features

- Answers university-related questions based on faculty guidelines and other documents
- Integrates seamlessly with Discord for easy access
- Utilizes LlamaIndex for efficient document indexing and retrieval
- Powered by Llama 3.1 for natural language processing

## Technologies Used

- Python
- LlamaIndex
- Llama 3.1
- Discord.py
- Docker
## Installation

The "prooven" way to install the chatbot is via ansible on a linux system. The
alternative way is to install the chatbot manually. Therefore, the playbook can be
used as a reference. The installation playbook is located in the `infrastructure`
directory.

### Step By Step Guide

1.  Install Ollama
   https://ollama.com/

2. Pull model
   ```bash
   ollama pull llama3.1:8b-instruct-q6_K
   ```

4. Install Python requirements
   ```bash
    pip install -r requirements.txt
   ```
   there may be missing some packages, just install them with pip
5. Install docker
   https://www.docker.com/

## Usage

### Environment Variables

The chatbot can be configured using environment variables. The following environment variables are available:

| Variable Name | Description                                  | Default Value    |
| ------------- | -------------------------------------------- | ---------------- |
| DISCORD_TOKEN | The Discord bot token                        | None             |
| DOCUMENTS_DIR | The directory where the documents are stored | ./data/documents |
| INDEX_DIR     | The directory where the index is stored      | ./data/index     |

Variables with no default value are required.

### Running the Chatbot
Start the redis data base by running following command:
```bash
docker compose up -d
```


To start the chatbot, run the following command:

```bash
python Bot_main.py
```

## Adding and Managing Sources for RAG

To manage the sources for the bot which can be used by the bot to answer question the RAG solution is choosed. All files
to include to the vector store must be placed in the **data/[course]** folder. For each course one **sources.json** is
needed. This file has the structure shown below:

```
{
  "sources": [
    {
      "priority": 1,
      "name": "Leitlinien Studium",
      "file": "Leitlinien.pdf",
      "description": "Informationen über die Studienordnung, Prüfungen",
      "web_link": "https://daten.de/Leitlinien.pdf"
    }
  ]
}
```

| Field       | Description                                                                                                                                                                                                             |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| priority    | Manipulates the "Similarity Score" used for ranking documents. Default value is **1**. This value represents an unchanged priority.<br>The Similarity Score (value from 0-1) is multiplied by this value.               |
| name        | Understandable human-readable identifier of the document. This name is displayed as source reference.                                                                                                                   |
| file        | Exact specification of the filename with file extension.                                                                                                                                                                |
| description | Brief description of the content of this file. When should one look into this file?                                                                                                                                     |
| web_link    | Link of the source. This link will be provided in chat. If the value is set to **"-"**, this document will not be included in the source reference and is thus a "hidden" source. This is useful for FAQs, for example. |

### Supported Document Types

https://docs.cloud.llamaindex.ai/llamaparse/features/supported_document_types

#### currently tested document types of the project:

.docx
.pdf
.xlsx
.txt
.csv

## Known Issues

- Error loading "fbgemm.dll" or one of its dependencies:
  - Solution: Install the Visual Studio Installer and the MSVC v143 VS Build Tools\
    Read more [here](https://github.com/pytorch/pytorch/issues/131662#issuecomment-2252589253)

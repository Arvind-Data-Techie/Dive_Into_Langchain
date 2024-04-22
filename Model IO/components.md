# Main Components of LangChain

LangChain offers standard, extensible interfaces, and external integrations for the following core components:

## Model I/O
Handles formatting and management of language model input and output.

## Prompts
Provides formatted inputs for Language Model (LM) generation, guiding the process.

## Chat Models
Interfaces for LMs utilizing chat messages as inputs and outputs, tailored for conversational interactions.

## LLMs
Interfaces for LMs processing plain text inputs and outputs.

## Retrieval
Enables interface with application-specific data retrieval, such as for Retrieval-Augmented Generation (RAG).

## Document Loaders
Loads data from various sources into documents for subsequent processing.

## Text Splitters
Transforms source documents to better fit specific application requirements.

## Embedding Models
Generates vector representations of text, facilitating natural language search capabilities.

## Vectorstores
Interfaces for specialized databases capable of natural language-based searches over unstructured data.

## Retrievers
Generic interfaces returning documents based on unstructured queries.

## Composition
Offers higher-level components combining other systems and LangChain primitives.

## Tools
Facilitates LLM interaction with external systems through standardized interfaces.

## Agents
Constructs that select appropriate tools based on predefined directives.

## Chains
Building block-style compositions of other runnable components, enhancing modularity and flexibility.

## Alfred - the gala agent

**Project Structure**:
- tools.py: provides auxiliary tools for the agent (Browser search, weather check, Hub Stats, ...)
- retriever.py: implements retrieval functions to support knowledge access (i.e., RAG)
- app.py: Integrates all components into a fully functional agent

- GuestbookTool.ipynb: Notebook with guestbook tool implementation (that retrieves info about guests). The code in this goes into retriever.py. (The notebook version was just for initial reference)

### To Run:
`python app.py`

### Possible Extensions:
- StreamLit chat app, to make the app/interface more interactive
- Improve the retriever to use a more sophisticated algorithm like sentence-transformers
- Implement a conversation memory so Alfred remembers previous interactions
- Combine with web search to get the latest information on unfamiliar guests
- Integrate multiple indexes to get more complete information from verified sources
- Extend the `retriever tool` to also return conversation starters based on guest's interests or background. 
import re

def parse_tags(query):
    """
    Parses a query for @tags.
    Returns the clean query and a list of tags.
    """
    tags = re.findall(r'@\w+', query)
    clean_query = re.sub(r'@\w+', '', query).strip()
    return clean_query, [t[1:] for t in tags]

def handle_interactive_query(agent, query, output_dir=None):
    """
    Handles user queries by parsing tags and routing to the right agent function.
    """
    clean_query, tags = parse_tags(query)
    
    if not tags:
        # Generic RAG query
        return agent.ask(query, output_dir=output_dir)
        
    responses = []
    for tag in tags:
        if tag.lower() in ["rag", "database", "knowledge"]:
            responses.append(f"**RAG Analysis:**\n{agent.ask(clean_query, output_dir=output_dir)}")
        elif tag.lower() == "anomaly":
            responses.append(f"**Anomaly Report:**\n{agent.analyze_behavior(output_dir=output_dir)}")
        else:
            # Assume tag is a penguin ID or custom query
            if hasattr(agent, 'tracks') and agent.tracks and tag in agent.tracks:
                responses.append(f"**Analysis for {tag}:**\n{agent.analyze_behavior(specific_id=tag, query=clean_query, output_dir=output_dir)}")
            else:
                # If tag is not a known penguin, ask using the tag as context
                rag_query = f"{tag} {clean_query}"
                responses.append(f"**Info on {tag}:**\n{agent.ask(rag_query, output_dir=output_dir)}")
                
    return "\n\n".join(responses)

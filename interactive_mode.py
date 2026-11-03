import re

def parse_tags(query):
    """
    Parses a query for @tags.
    Returns the clean query and a list of tags.
    """
    # Emails are ordinary question text; repeated tags must not repeat inference.
    pattern = r'(?<![\w@])@(\w+)'
    tags = list(dict.fromkeys(re.findall(pattern, query)))
    clean_query = re.sub(pattern, '', query).strip()
    return clean_query, tags

def handle_interactive_query(agent, query, output_dir=None):
    """
    Handles user queries by parsing tags and routing to the right agent function.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Enter a nonempty question.")
    clean_query, tags = parse_tags(query)
    
    if not tags:
        # Generic RAG query
        return agent.ask(query, output_dir=output_dir)
        
    responses = []
    processed = set()
    for tag in tags:
        operation = "rag" if tag.lower() in {"rag", "database", "knowledge"} else tag.lower() if tag.lower() == "anomaly" else tag
        if operation in processed:
            continue
        processed.add(operation)
        if tag.lower() in ["rag", "database", "knowledge"]:
            if not clean_query:
                raise ValueError("Enter a scientific question after the RAG tag.")
            responses.append(f"**RAG Analysis:**\n{agent.ask(clean_query, output_dir=output_dir)}")
        elif tag.lower() == "anomaly":
            if output_dir is None or not getattr(agent, "tracks", None):
                responses.append("No validated video analysis is available. Run tracking before using @anomaly.")
            else:
                responses.append(f"**Anomaly Report:**\n{agent.analyze_behavior(query=clean_query, output_dir=output_dir)}")
        else:
            # Assume tag is a penguin ID or custom query
            if output_dir is not None and getattr(agent, 'tracks', None) and tag in agent.tracks:
                responses.append(f"**Analysis for {tag}:**\n{agent.analyze_behavior(specific_id=tag, query=clean_query, output_dir=output_dir)}")
            else:
                # If tag is not a known penguin, ask using the tag as context
                rag_query = f"{tag} {clean_query}"
                responses.append(f"**Info on {tag}:**\n{agent.ask(rag_query, output_dir=output_dir)}")
                
    return "\n\n".join(responses)

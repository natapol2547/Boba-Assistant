import re, requests
from typing import List, Dict, Union

def api_search(query = "", num_results = 5, time_period = "", region = "") -> List[Dict[str, Union[str, None]]]:
    page_operator_matches = re.search(r'page:(\S+)', query)
    query_url = None

    if page_operator_matches:
        query_url = page_operator_matches.group(1)

    url = None
    if query_url:
        url = f'https://ddg-webapp-aagd.vercel.app/url_to_text?url={query_url}'
    else:
        url = f'https://ddg-webapp-aagd.vercel.app/search?' \
              f'max_results={num_results}&q={query}' \
              f'{f"&time={time_period}" if time_period else ""}' \
              f'{f"&region={region}" if region else ""}'

    response = requests.get(url)
    results = response.json()
    unformatted = [{"body": result["body"], "href": result["href"], "title": result["title"]} for result in results]
    counter = 1
    formattedResults = ""
    for result in unformatted:
        formattedResults += f"{result['body']}\nFrom website: {result['href']}\n\n"
        counter += 1
    return formattedResults
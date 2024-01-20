import json


# Fake weather API (example provided in OpenAI API docs)
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps(
            {"location": location, "temperature": "10", "unit": "celsius"}
        )
    elif "san francisco" in location.lower():
        return json.dumps(
            {"location": location, "temperature": "72", "unit": "fahrenheit"}
        )
    elif "rotterdam" in location.lower():
        return json.dumps({"location": location, "temperature": "1", "unit": "celsius"})
    elif "amsterdam" in location.lower():
        return json.dumps(
            {"location": location, "temperature": "-2", "unit": "celsius"}
        )
    elif "zoetermeer" in location.lower():
        return json.dumps({"location": location, "temperature": "3", "unit": "celsius"})
    else:
        return json.dumps(
            {"location": location, "temperature": "22", "unit": "celsius"}
        )

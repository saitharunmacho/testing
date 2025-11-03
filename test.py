import requests
import json

DATABRICKS_HOST = ""
DATABRICKS_TOKEN = ""
GENIE_SPACE_ENDPOINT = ""


print(f'starting the conversation \n')
request_data ={"content": "list out top 10 Business Unit based on their revenue"}
response = requests.post(f"{DATABRICKS_HOST}/api/2.0/genie/spaces/{GENIE_SPACE_ENDPOINT}/start-conversation", headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}"}, json=request_data)
print(response.json())
conversation_id = response.json()['conversation_id']
message_id = response.json()['message_id']



print(f'message_id: {message_id}\n')
print(f'conversation_id: {conversation_id}\n')






print(f'creating the conversation message \n')
request_data ={"content": "list out top 10 Business Unit based on their revenue"}
response = requests.post(f"{DATABRICKS_HOST}/api/2.0/genie/spaces/{GENIE_SPACE_ENDPOINT}/conversations/{conversation_id}/messages", headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}"}, json=request_data)
print(f'the response is: {response.json()}\n')




print(f'listing the conversations in the genie space \n')
list_of_conversations_response = requests.get(f"{DATABRICKS_HOST}/api/2.0/genie/spaces/{GENIE_SPACE_ENDPOINT}/conversations/{conversation_id}/messages", headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}"})
print(f'list_of_conversations_response: {list_of_conversations_response.json()}\n')



print(f'getting the message along with the attachments \n')







# print(f'listing the conversations in the genie space \n')
# list_of_conversations_response = requests.get(f'{DATABRICKS_HOST}/api/2.0/genie/spaces/{GENIE_SPACE_ENDPOINT}/conversations', headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}"})
# print(f'list_of_conversations_response: {list_of_conversations_response.json()}\n')


# print(f'now gettin the list conversation messages which is expected to have the attachment id \n')
# response_with_attachment_id = requests.get(f'{DATABRICKS_HOST}/api/2.0/genie/spaces/{GENIE_SPACE_ENDPOINT}/conversations/{conversation_id}/messages', headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}"})
# print(f'the response with the attachment id is: {response_with_attachment_id.json()}\n')















# print(f'getting the attachment id \n')
# attachment_id_response = requests.get(f'{DATABRICKS_HOST}/api/2.0/genie/spaces/{GENIE_SPACE_ENDPOINT}/conversations/{conversation_id}/messages/{message_id}', headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}"})
# print(f'attachment_id_response: {attachment_id_response.json()}\n')

# attachment_dict = json.loads(attachment_id_response.text)
# attachment_id = attachment_dict.get('id')
# print(f'attachment_id: {attachment_id}\n')



# print(f'getting the query result \n')
# query_result_response = requests.get(f'{DATABRICKS_HOST}/api/2.0/genie/spaces/{GENIE_SPACE_ENDPOINT}/conversations/{conversation_id}/messages/{message_id}/query-result/{attachment_id}', headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}"})
# print(f'query_result_response: {query_result_response.json()}\n')


# print(f'listing conversations messages \n')
# list_of_conversations_messages_response = requests.get(f'{DATABRICKS_HOST}/api/2.0/genie/spaces/{GENIE_SPACE_ENDPOINT}/conversations/{conversation_id}/messages', headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}"})
# print(f'list_of_conversations_messages_response: {list_of_conversations_messages_response.json()}\n')




# attachment_id = response.json()['attachments'][0]['id']
# print("listing conversations\n")
# list_of_conversations_response = requests.get(f'{DATABRICKS_HOST}/api/2.0/genie/spaces/{GENIE_SPACE_ENDPOINT}/conversations/{conversation_id}/messages/{message_id}/query-result/{attachment_id}', headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}"})
# print(f'list_of_conversations_response: {list_of_conversations_response.json()}\n')



# attachment_id = list_of_conversations_response.json()['attachments'][0]['id']
# print(f'attachment_id: {attachment_id}\n')

# message_id = response.json()['message_id']
# conversation_id = response.json()['conversation_id']
# # attachment_id = response.json()['attachments'][0]['id']

# # print(f'getting the attachents\n')
# # attachment_response = requests.get(f'/api/2.0/genie/spaces/{GENIE_SPACE_ENDPOINT}/conversations/{conversation_id}/messages/{message_id}/attachments/{attachment_id}/query-result')
# # print(attachment_response.json())

# print(f'getting the message\n')
# message_response = requests.get(f'/api/2.0/genie/spaces/{GENIE_SPACE_ENDPOINT}/conversations/{conversation_id}/messages', headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}"})
# print(message_response.json())




# print(f'getting the message attachments\n')
# message_attachments_response = requests.get(f'/api/2.0/genie/spaces/{GENIE_SPACE_ENDPOINT}/conversations/{conversation_id}/messages/{message_id}/attachments/{attachment_id}/query-result', headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}"})
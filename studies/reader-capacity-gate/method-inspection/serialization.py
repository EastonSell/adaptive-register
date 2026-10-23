"""Frozen profile/glossary/source serialization shared by all proposed arms."""
SERIALIZATION_VERSION = 'reader-profile-chat-v1'
INSTRUCTION = 'Rewrite faithfully. Explain glossary concepts not listed as known.'
def user_content(record, profile, concepts):
    known = '; '.join(x['concept_id']+'='+x['meaning'] for x in profile['known_meanings'])
    glossary = '; '.join(cid+'='+concepts[cid]['definition'] for cid in record['concept_ids'])
    return INSTRUCTION+'\nKnown: '+known+'\nGlossary: '+glossary+'\nSource: '+record['source']
def messages(record,profile,concepts,reference=None):
    result=[{'role':'user','content':user_content(record,profile,concepts)}]
    if reference is not None:result.append({'role':'assistant','content':reference['text']})
    return result

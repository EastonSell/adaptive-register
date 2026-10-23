"""Conservative frozen-denominator semantic aggregation; does not judge language."""
from .schedule import need


def blank_rating(case):
    return {'output_id':case['output_id'],'source_id':case['source_id'],'profile_id':case['profile_id'],
            'adjudication_complete':False,'fluency':'unscored','task_following':'unscored','fluency_rationale':'','task_following_rationale':'',
            'propositions':[{'proposition_id':p,'status':'unscored','span':'','rationale':''} for p in case['proposition_ids']],
            'unknown_concepts':[{'concept_id':c,'status':'unscored','span':'','rationale':''} for c in case['unknown_concept_ids']],
            'unsupported_assertions':None,'known_explanations':[],'adjudicator':'','adjudicator_kind':'',
            'disagreement_resolved':False,'notes':''}


def aggregate(cases,ratings,*,outputs):
    need(len(cases)==24 and len({x['output_id'] for x in cases})==24,'exactly 24 unique frozen outputs required')
    lookup={x['output_id']:x for x in cases};rows={}
    for row in ratings:
        oid=row['output_id'];need(oid in lookup and oid not in rows,'unknown/duplicate rating output')
        case=lookup[oid]
        output=outputs.get(oid)
        need(output is not None and output.get('output_id')==oid and output.get('status')=='returned' and isinstance(output.get('scoring_text'),str),'rating has no retained returned output')
        text=output['scoring_text']
        need(all(row[k]==case[k] for k in ('source_id','profile_id')),'rating linkage changed')
        for key,idkey,wanted,allowed in [('propositions','proposition_id',case['proposition_ids'],{'preserved','omitted','contradicted','uncertain','unscored'}),('unknown_concepts','concept_id',case['unknown_concept_ids'],{'supported','inaccurate','absent_or_unglossed','uncertain','unscored'})]:
            items=row[key];need([x[idkey] for x in items]==wanted,'rating proposition/concept mapping changed')
            for x in items:
                need(x['status'] in allowed,'invalid semantic status')
                if x['status'] in ('preserved','supported'):need(isinstance(x['span'],str) and bool(x['span'].strip()) and isinstance(x['rationale'],str) and bool(x['rationale'].strip()),'passing judgment requires evidence span and rationale')
                if x['status'] in ('preserved','supported'):need(x['span'] in text,'evidence span is absent from actual generated output')
        need(row['fluency'] in ('pass','fail','uncertain','unscored') and row['task_following'] in ('pass','fail','uncertain','unscored'),'invalid presentation status')
        need(type(row['adjudication_complete']) is bool,'strict adjudication-complete boolean required')
        need(row['unsupported_assertions'] is None or isinstance(row['unsupported_assertions'],list),'unsupported assertion record invalid')
        if row['adjudication_complete']:
            need(bool(row['adjudicator'].strip()) and row['adjudicator_kind'] in ('AI','human') and row['disagreement_resolved'] is True,'completed rating requires adjudicator identity/kind and resolved disagreements')
            need(row['unsupported_assertions'] is not None and all(x['status']!='unscored' and bool(x['rationale'].strip()) for x in row['propositions']+row['unknown_concepts']),'completed rating has unscored semantic fields')
            need(row['fluency']!='unscored' and row['task_following']!='unscored' and bool(row.get('fluency_rationale','').strip()) and bool(row.get('task_following_rationale','').strip()),'completed presentation rating/rationale missing')
        for assertion in row['unsupported_assertions'] or []:
            need(isinstance(assertion,dict) and bool(assertion.get('span','')) and assertion['span'] in text and bool(assertion.get('rationale','').strip()),'unsupported assertion requires actual output span/rationale')
        rows[oid]=row
    result={'denominator':24,'fluent_truth_complete':0,'unknown_support':0,'truth_safe_and_supported':0,
            'truth_complete':0,'truth_safe':0,'fluency_pass':0,'task_following_pass':0,
            'proposition_status_counts':{k:0 for k in ('preserved','omitted','contradicted','uncertain','unscored')},
            'unknown_status_counts':{k:0 for k in ('supported','inaccurate','absent_or_unglossed','uncertain','unscored')},
            'outputs_with_unsupported_assertions':0,'missing_ratings':24-len(rows),'unscored_or_incomplete':0,'case_results':[]}
    for case in cases:
        row=rows.get(case['output_id']);truth=fluency=support=safe=False
        if row:
            complete=row['adjudication_complete']
            truth=complete and all(x['status']=='preserved' for x in row['propositions'])
            fluency=truth and row['fluency']=='pass' and row['task_following']=='pass'
            support=complete and all(x['status']=='supported' for x in row['unknown_concepts'])
            safe=truth and row['unsupported_assertions']==[]
            result['outputs_with_unsupported_assertions']+=int(bool(row['unsupported_assertions']))
            result['unscored_or_incomplete']+=int(not complete)
            result['truth_complete']+=int(truth);result['truth_safe']+=int(safe)
            result['fluency_pass']+=int(complete and row['fluency']=='pass')
            result['task_following_pass']+=int(complete and row['task_following']=='pass')
            for p in row['propositions']:result['proposition_status_counts'][p['status']]+=1
            for c in row['unknown_concepts']:result['unknown_status_counts'][c['status']]+=1
        else:
            result['proposition_status_counts']['unscored']+=len(case['proposition_ids'])
            result['unknown_status_counts']['unscored']+=len(case['unknown_concept_ids'])
        result['fluent_truth_complete']+=int(fluency);result['unknown_support']+=int(support)
        result['truth_safe_and_supported']+=int(safe and support)
        result['case_results'].append({'output_id':case['output_id'],'truth_complete':bool(truth),'truth_safe':bool(safe),'fluent_truth_complete':bool(fluency),'unknown_support':bool(support),'truth_safe_and_supported':bool(safe and support)})
    result['adjudicated_outputs']=sum(row['adjudication_complete'] for row in rows.values())
    result['semantic_status']='unmeasured' if result['adjudicated_outputs']==0 else 'complete' if result['adjudicated_outputs']==24 else 'partially_measured_conservative_missing_failures'
    result['capacity_gate_passed']=None if result['adjudicated_outputs']==0 else result['fluent_truth_complete']>=18 and result['unknown_support']>=12
    result['interpretation']='A-only pragmatic development-set feasibility; no comparative-arm efficacy, human-comprehension or test-set conclusion.'
    return result

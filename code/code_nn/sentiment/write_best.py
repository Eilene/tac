# coding=utf-8

import xml.dom.minidom
import os
import shutil
import numpy as np


# 输出best.xml
def write_best_file(file_info, no, output_dir):
    doc = xml.dom.minidom.Document()
    best = doc.createElement('belief_sentiment_doc')
    best.setAttribute('id', "tree-56acee9a00000000000000"+str(no))
    doc.appendChild(best)
    st = doc.createElement('sentiment_annotations')
    best.appendChild(st)

    # entity
    if 'entity' in file_info:
        entities = doc.createElement('entities')
        st.appendChild(entities)
        for i in range(len(file_info['entity'])):
            entity = doc.createElement('entity')
            entity.setAttribute('ere_id', str(file_info['entity'][i]['entity_mention_id']))
            entity.setAttribute('offset', str(file_info['entity'][i]['entity_mention_offset']))
            entity.setAttribute('length', str(file_info['entity'][i]['entity_mention_length']))
            entities.appendChild(entity)

            text = doc.createElement('text')
            text_text = doc.createTextNode(str(file_info['entity'][i]['entity_mention_text']))
            text.appendChild(text_text)
            entity.appendChild(text)

            sentiments = doc.createElement('sentiments')
            entity.appendChild(sentiments)

            sentiment = doc.createElement('sentiment')
            sentiment.setAttribute('polarity', str(file_info['entity'][i]['predict_polarity']))  # 改下，现在这是列表，改成neg，pos
            sentiment.setAttribute('sarcasm', 'no')
            sentiments.appendChild(sentiment)

            if file_info['entity'][i]['predict_polarity'] != 'none':
                source = doc.createElement('source')
                if 'predict_source_id' in file_info['entity'][i]:  # 即非空
                    source.setAttribute('ere_id', str(file_info['entity'][i]['predict_source_id']))
                    source.setAttribute('offset', str(file_info['entity'][i]['predict_source_offset']))
                    source.setAttribute('length', str(file_info['entity'][i]['predict_source_length']))
                    source_text = doc.createTextNode(file_info['entity'][i]['predict_source_text'])
                    source.appendChild(source_text)

                sentiment.appendChild(source)

    # relations
    if 'relation' in file_info:
        relations = doc.createElement('relations')
        st.appendChild(relations)
        for i in range(len(file_info['relation'])):
            relation = doc.createElement('relation')
            relation.setAttribute('ere_id', str(file_info['relation'][i]['relation_mention_id']))
            relations.appendChild(relation)

            if int(file_info['relation'][i]['trigger_length']) != 0:
                trigger = doc.createElement('trigger')
                trigger_text = doc.createTextNode(str(file_info['relation'][i]['trigger_text']))
                trigger.setAttribute('offset', str(int(file_info['relation'][i]['trigger_offset'])))
                trigger.setAttribute('length', str(int(file_info['relation'][i]['trigger_length'])))
                trigger.appendChild(trigger_text)
                relation.appendChild(trigger)

            rsentiments = doc.createElement('sentiments')
            relation.appendChild(rsentiments)

            rsentiment = doc.createElement('sentiment')
            rsentiment.setAttribute('polarity', str(file_info['relation'][i]['predict_polarity']))
            rsentiment.setAttribute('sarcasm', 'no')
            rsentiments.appendChild(rsentiment)

            if file_info['relation'][i]['predict_polarity'] != 'none':
                rsource = doc.createElement('source')
                if 'predict_source_id' in file_info['relation'][i]:  # 即非空
                    rsource.setAttribute('ere_id', str(file_info['relation'][i]['predict_source_id']))
                    rsource.setAttribute('offset', str(file_info['relation'][i]['predict_source_offset']))
                    rsource.setAttribute('length', str(file_info['relation'][i]['predict_source_length']))
                    rsource_text = doc.createTextNode(file_info['relation'][i]['predict_source_text'])
                    rsource.appendChild(rsource_text)
                rsentiment.appendChild(rsource)

    # events
    if 'event' in file_info:
        events = doc.createElement('events')
        st.appendChild(events)
        for i in range(len(file_info['event'])):
            event = doc.createElement('event')
            event.setAttribute('ere_id', str(file_info['event'][i]['event_mention_id']))
            events.appendChild(event)

            etrigger = doc.createElement('trigger')
            etrigger_text = doc.createTextNode(str(file_info['event'][i]['trigger_text']))
            event.setAttribute('offset', str(file_info['event'][i]['trigger_offset']))
            event.setAttribute('length', str(file_info['event'][i]['trigger_length']))
            etrigger.appendChild(etrigger_text)
            event.appendChild(etrigger)

            esentiments = doc.createElement('sentiments')
            event.appendChild(esentiments)

            esentiment = doc.createElement('sentiment')
            esentiment.setAttribute('polarity', str(file_info['event'][i]['predict_polarity']))
            esentiment.setAttribute('sarcasm', 'no')
            esentiments.appendChild(esentiment)

            if file_info['event'][i]['predict_polarity'] != 'none':
                esource = doc.createElement('source')
                if 'predict_source_id' in file_info['event'][i]:  # 即非空
                    esource.setAttribute('ere_id', str(file_info['event'][i]['predict_source_id']))
                    esource.setAttribute('offset', str(file_info['event'][i]['predict_source_offset']))
                    esource.setAttribute('length', str(file_info['event'][i]['predict_source_length']))
                    esource_text = doc.createTextNode(file_info['event'][i]['predict_source_text'])
                    esource.appendChild(esource_text)
                esentiment.appendChild(esource)

    f = open(output_dir + file_info['filename'] + '.best.xml', 'w')
    # f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))
    f.write(doc.toprettyxml())
    f.close()


def write_best_files(results, output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    for i in range(len(results)):
        write_best_file(results[i], i, output_dir)

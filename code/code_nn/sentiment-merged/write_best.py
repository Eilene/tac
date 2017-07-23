# coding=utf-8

import xml.dom.minidom


# 输出best.xml
def write_best(file_info, no):
    doc = xml.dom.minidom.Document()
    best = doc.createElement('belief_sentiment_doc')
    best.setAttribute('id', "tree-56acee9a00000000000000"+str(no))
    doc.appendChild(best)
    st = doc.createElement('sentiment_annotations')
    best.appendChild(st)

    # entity
    if file_info['entity'] is not None:
        entities = doc.createElement('entities')
        st.appendChild(entities)
        for i in range(len(file_info['entity'])):
            entity = doc.createElement('entity')
            entity.setAttribute('ere_id', file_info['entity'][i]['mention_id'])
            entity.setAttribute('offset', str(file_info['entity'][i]['mention_offset']))
            entity.setAttribute('length', str(len(file_info['entity'][i]['mention_text'])))
            entities.appendChild(entity)

            text = doc.createElement('text')
            text_text = doc.createTextNode(file_info['entity'][i]['mention_text'])
            text.appendChild(text_text)
            entity.appendChild(text)

            sentiments = doc.createElement('sentiments')
            entity.appendChild(sentiments)

            sentiment = doc.createElement('sentiment')
            sentiment.setAttribute('polarity', file_info['entity'][i]['predict_polarity'])
            sentiment.setAttribute('sarcasm', 'no')
            sentiments.appendChild(sentiment)

            if file_info['entity'][i]['predict_polarity'] != 'none':
                source = doc.createElement('source')
                if file_info['entity'][i]['source'] is not None:
                    source.setAttribute('ere_id', str(file_info['entity'][i]['source']['ere_id']))
                    source.setAttribute('offset', str(file_info['entity'][i]['source']['offset']))
                    source.setAttribute('length', str(file_info['entity'][i]['source']['length']))
                sentiment.appendChild(source)

    # relations
    if file_info['relation'] is not None:
        relations = doc.createElement('relations')
        st.appendChild(relations)
        for i in range(len(file_info['relation'])):
            relation = doc.createElement('relation')
            relation.setAttribute('ere_id', file_info['relation'][i]['relation_mention_id'])
            relations.appendChild(relation)

            trigger = doc.createElement('trigger')
            trigger_text = doc.createTextNode(file_info['relation'][i]['trigger_text'])
            relation.setAttribute('offset', str(file_info['relation'][i]['trigger_offset']))
            relation.setAttribute('length', str(file_info['relation'][i]['trigger_length']))
            trigger.appendChild(trigger_text)
            relation.appendChild(trigger)

            rsentiments = doc.createElement('sentiments')
            relation.appendChild(rsentiments)

            rsentiment = doc.createElement('sentiment')
            rsentiment.setAttribute('polarity', file_info['relation'][i]['predict_polarity'])
            rsentiment.setAttribute('sarcasm', 'no')
            rsentiments.appendChild(rsentiment)

            if file_info['relation'][i]['predict_polarity'] != 'none':
                rsource = doc.createElement('source')
                if file_info['relation'][i]['source'] is not None:
                    rsource.setAttribute('ere_id', str(file_info['relation'][i]['source']['ere_id']))
                    rsource.setAttribute('offset', str(file_info['relation'][i]['source']['offset']))
                    rsource.setAttribute('length', str(file_info['relation'][i]['source']['length']))
                rsentiment.appendChild(rsource)

    # events
    if file_info['event'] is not None:
        events = doc.createElement('events')
        st.appendChild(events)
        for i in range(len(file_info['event'])):
            event = doc.createElement('event')
            event.setAttribute('ere_id', file_info['event'][i]['event_mention_id'])
            events.appendChild(event)

            etrigger = doc.createElement('trigger')
            etrigger_text = doc.createTextNode(file_info['event'][i]['trigger_text'])
            event.setAttribute('offset', str(file_info['event'][i]['trigger_offset']))
            event.setAttribute('length', str(file_info['event'][i]['trigger_length']))
            etrigger.appendChild(etrigger_text)
            event.appendChild(etrigger)

            esentiments = doc.createElement('sentiments')
            event.appendChild(esentiments)

            esentiment = doc.createElement('sentiment')
            esentiment.setAttribute('polarity', file_info['event'][i]['predict_polarity'])
            esentiment.setAttribute('sarcasm', 'no')
            esentiments.appendChild(esentiment)

            if file_info['event'][i]['predict_polarity'] != 'none':
                esource = doc.createElement('source')
                if file_info['event'][i]['source'] is not None:
                    esource.setAttribute('ere_id', str(file_info['event'][i]['source']['ere_id']))
                    esource.setAttribute('offset', str(file_info['event'][i]['source']['offset']))
                    esource.setAttribute('length', str(file_info['event'][i]['source']['length']))
                esentiment.appendChild(esource)

    f = open('output/' + file_info['filename'] + '.best.xml', 'w')
    # f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))
    f.write(doc.toprettyxml())
    f.close()


def write_best_files(results):
    for i in range(len(results)):
        write_best(results[i], i)

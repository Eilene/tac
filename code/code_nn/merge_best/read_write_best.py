# coding=utf-8
import xml.dom.minidom


import sys
reload(sys)
sys.setdefaultencoding("utf-8")


def read_st_file(filepath):
    file_record_dict = {}

    st_file = xml.dom.minidom.parse(filepath)
    root = st_file.documentElement
    sentiment_annotations = root.getElementsByTagName('sentiment_annotations')
    sentiment_annotations = sentiment_annotations[0]

    entity_list = sentiment_annotations.getElementsByTagName('entity')
    if len(entity_list) != 0:
        file_record_dict['entity'] = []
        for i in range(len(entity_list)):
            record = {}
            record['ere_id'] = entity_list[i].getAttribute('ere_id')
            record['offset'] = entity_list[i].getAttribute('offset')
            record['length'] = entity_list[i].getAttribute('length')
            text_em = entity_list[i].getElementsByTagName('text')
            text_em = text_em[0]
            record['text'] = text_em.firstChild.data
            be_em = entity_list[i].getElementsByTagName('sentiment')
            be_em = be_em[0]
            polarity = be_em.getAttribute('polarity')
            record['polarity'] = polarity
            record['sarcasm'] = be_em.getAttribute('sarcasm')
            if polarity != 'none':
                source_em = be_em.getElementsByTagName('source')
                if len(source_em) != 0:
                    source_em = source_em[0]
                    record['source_ere_id'] = source_em.getAttribute('ere_id')
                    record['source_offset'] = source_em.getAttribute('offset')
                    record['source_length'] = source_em.getAttribute('length')
                    record['source_text'] = source_em.firstChild.data
            file_record_dict['entity'].append(record)

    relation_list = sentiment_annotations.getElementsByTagName('relation')
    if len(relation_list) != 0:
        file_record_dict['relation'] = []
        for i in range(len(relation_list)):
            record = {}
            record['ere_id'] = relation_list[i].getAttribute('ere_id')
            trigger = relation_list[i].getElementsByTagName('trigger')
            if len(trigger) != 0:
                trigger = trigger[0]
                record['trigger_offset'] = trigger.getAttribute('offset')
                record['trigger_length'] = trigger.getAttribute('length')
                record['trigger_text'] = trigger.firstChild.data
            be_em = entity_list[i].getElementsByTagName('sentiment')
            be_em = be_em[0]
            polarity = be_em.getAttribute('polarity')
            record['polarity'] = polarity
            record['sarcasm'] = be_em.getAttribute('sarcasm')
            if polarity != 'none':
                source_em = be_em.getElementsByTagName('source')
                if len(source_em) != 0:
                    source_em = source_em[0]
                    record['source_ere_id'] = source_em.getAttribute('ere_id')
                    record['source_offset'] = source_em.getAttribute('offset')
                    record['source_length'] = source_em.getAttribute('length')
                    record['source_text'] = source_em.firstChild.data
            file_record_dict['relation'].append(record)

    event_list = sentiment_annotations.getElementsByTagName('event')
    if len(event_list) != 0:
        file_record_dict['event'] = []
        for i in range(len(event_list)):
            record = {}
            record['ere_id'] = event_list[i].getAttribute('ere_id')
            trigger = event_list[i].getElementsByTagName('trigger')
            trigger = trigger[0]
            record['trigger_offset'] = trigger.getAttribute('offset')
            record['trigger_length'] = trigger.getAttribute('length')
            record['trigger_text'] = trigger.firstChild.data
            be_em = event_list[i].getElementsByTagName('sentiment')
            be_em = be_em[0]
            polarity = be_em.getAttribute('polarity')
            record['polarity'] = polarity
            record['sarcasm'] = be_em.getAttribute('sarcasm')
            if polarity != 'none':
                source_em = be_em.getElementsByTagName('source')
                if len(source_em) != 0:
                    source_em = source_em[0]
                    record['source_ere_id'] = source_em.getAttribute('ere_id')
                    record['source_offset'] = source_em.getAttribute('offset')
                    record['source_length'] = source_em.getAttribute('length')
                    record['source_text'] = source_em.firstChild.data
            file_record_dict['event'].append(record)

    return file_record_dict


def read_be_file(filepath):
    file_record_dict = {}

    be_file = xml.dom.minidom.parse(filepath)
    root = be_file.documentElement
    belief_annotations = root.getElementsByTagName('belief_annotations')
    belief_annotations = belief_annotations[0]

    relation_list = belief_annotations.getElementsByTagName('relation')
    if len(relation_list) != 0:
        file_record_dict['relation'] = []
        for i in range(len(relation_list)):
            record = {}
            record['ere_id'] = relation_list[i].getAttribute('ere_id')
            trigger = relation_list[i].getElementsByTagName('trigger')
            if len(trigger) != 0:
                trigger = trigger[0]
                record['trigger_offset'] = trigger.getAttribute('offset')
                record['trigger_length'] = trigger.getAttribute('length')
                record['trigger_text'] = trigger.firstChild.data
            be_em = relation_list[i].getElementsByTagName('belief')
            be_em = be_em[0]
            record['type'] = be_em.getAttribute('type')
            record['polarity'] = be_em.getAttribute('polarity')
            record['sarcasm'] = be_em.getAttribute('sarcasm')
            source_em = be_em.getElementsByTagName('source')
            if len(source_em) != 0:
                source_em = source_em[0]
                record['source_ere_id'] = source_em.getAttribute('ere_id')
                record['source_offset'] = source_em.getAttribute('offset')
                record['source_length'] = source_em.getAttribute('length')
                record['source_text'] = source_em.firstChild.data
            file_record_dict['relation'].append(record)

    event_list = belief_annotations.getElementsByTagName('event')
    if len(event_list) != 0:
        file_record_dict['event'] = []
        for i in range(len(event_list)):
            record = {}
            record['ere_id'] = event_list[i].getAttribute('ere_id')
            trigger = event_list[i].getElementsByTagName('trigger')
            trigger = trigger[0]
            record['trigger_offset'] = trigger.getAttribute('offset')
            record['trigger_length'] = trigger.getAttribute('length')
            record['trigger_text'] = trigger.firstChild.data
            be_em = event_list[i].getElementsByTagName('belief')
            be_em = be_em[0]
            record['type'] = be_em.getAttribute('type')
            record['polarity'] = be_em.getAttribute('polarity')
            record['sarcasm'] = be_em.getAttribute('sarcasm')
            source_em = be_em.getElementsByTagName('source')
            if len(source_em) != 0:
                source_em = source_em[0]
                record['source_ere_id'] = source_em.getAttribute('ere_id')
                record['source_offset'] = source_em.getAttribute('offset')
                record['source_length'] = source_em.getAttribute('length')
                record['source_text'] = source_em.firstChild.data
            # argments
            arg_list = event_list[i].getElementsByTagName('arg')
            if len(arg_list) != 0:
                em_args = []
                for em_arg in arg_list:
                    arg_record = {}
                    arg_record['ere_id'] = em_arg.getAttribute('ere_id')
                    arg_record['offset'] = em_arg.getAttribute('offset')
                    arg_record['length'] = em_arg.getAttribute('length')
                    text_em = em_arg.getElementsByTagName('text')
                    arg_record['text'] = text_em[0].firstChild.data
                    em_args.append(arg_record)
                    # belief部分先直接用event的
                record['em_args'] = em_args
            file_record_dict['event'].append(record)

    return file_record_dict


def write_best_file(st_file_record_dict, be_file_record_dict, filepath, no):

    doc = xml.dom.minidom.Document()
    best = doc.createElement('belief_sentiment_doc')
    best.setAttribute('id', "tree-56acee9a00000000000000"+str(no))
    doc.appendChild(best)
    
    # ** belief **
    be = doc.createElement('belief_annotations')
    best.appendChild(be)

    # relations
    if 'relation' in be_file_record_dict:
        relations = doc.createElement('relations')
        be.appendChild(relations)
        for i in range(len(be_file_record_dict['relation'])):
            relation = doc.createElement('relation')
            relation.setAttribute('ere_id', str(be_file_record_dict['relation'][i]['ere_id']))
            relations.appendChild(relation)

            if 'trigger_length' in st_file_record_dict['relation'][i]:
                trigger = doc.createElement('trigger')
                trigger_text = doc.createTextNode(str(be_file_record_dict['relation'][i]['trigger_text']))
                trigger.setAttribute('offset', str(int(be_file_record_dict['relation'][i]['trigger_offset'])))
                trigger.setAttribute('length', str(int(be_file_record_dict['relation'][i]['trigger_length'])))
                trigger.appendChild(trigger_text)
                relation.appendChild(trigger)

            rbeliefs = doc.createElement('beliefs')
            relation.appendChild(rbeliefs)

            rbelief = doc.createElement('belief')
            rbelief.setAttribute('type', str(be_file_record_dict['relation'][i]['type']))
            rbelief.setAttribute('polarity', str(be_file_record_dict['relation'][i]['polarity']))
            rbelief.setAttribute('sarcasm', str(be_file_record_dict['relation'][i]['polarity']))
            rbeliefs.appendChild(rbelief)

            if 'source_ere_id' in be_file_record_dict['relation'][i]:  # 即非空
                rsource = doc.createElement('source')
                rsource.setAttribute('ere_id', str(be_file_record_dict['relation'][i]['source_ere_id']))
                rsource.setAttribute('offset', str(be_file_record_dict['relation'][i]['source_offset']))
                rsource.setAttribute('length', str(be_file_record_dict['relation'][i]['source_length']))
                rsource_text = doc.createTextNode(be_file_record_dict['relation'][i]['source_text'])
                rsource.appendChild(rsource_text)
                rbelief.appendChild(rsource)

    # events
    if 'event' in be_file_record_dict:
        events = doc.createElement('events')
        be.appendChild(events)
        for i in range(len(be_file_record_dict['event'])):
            event = doc.createElement('event')
            event.setAttribute('ere_id', str(be_file_record_dict['event'][i]['ere_id']))
            events.appendChild(event)

            etrigger = doc.createElement('trigger')
            etrigger_text = doc.createTextNode(str(be_file_record_dict['event'][i]['trigger_text']))
            etrigger.setAttribute('offset', str(be_file_record_dict['event'][i]['trigger_offset']))
            etrigger.setAttribute('length', str(be_file_record_dict['event'][i]['trigger_length']))
            etrigger.appendChild(etrigger_text)
            event.appendChild(etrigger)

            ebeliefs = doc.createElement('beliefs')
            event.appendChild(ebeliefs)

            ebelief = doc.createElement('belief')
            ebelief.setAttribute('type', str(be_file_record_dict['event'][i]['type']))
            ebelief.setAttribute('polarity', 'pos')
            ebelief.setAttribute('sarcasm', 'no')
            ebeliefs.appendChild(ebelief)

            if 'source_ere_id' in be_file_record_dict['event'][i]:  # 即非空
                esource = doc.createElement('source')
                esource.setAttribute('ere_id', str(be_file_record_dict['event'][i]['source_ere_id']))
                esource.setAttribute('offset', str(be_file_record_dict['event'][i]['source_offset']))
                esource.setAttribute('length', str(be_file_record_dict['event'][i]['source_length']))
                esource_text = doc.createTextNode(be_file_record_dict['event'][i]['source_text'])
                esource.appendChild(esource_text)
                ebelief.appendChild(esource)

            # argments
            if 'em_args' in be_file_record_dict['event'][i]:
                arguments = doc.createElement('arguments')
                event.appendChild(arguments)
                for em_arg in be_file_record_dict['event'][i]['em_args']:
                    arg = doc.createElement('arg')
                    arguments.appendChild(arg)
                    arg.setAttribute('ere_id', str(em_arg['ere_id']))
                    arg.setAttribute('offset', str(em_arg['offset']))
                    arg.setAttribute('length', str(em_arg['length']))
                    text = doc.createElement('text')
                    arg.appendChild(text)
                    text_text = doc.createTextNode(em_arg['text'])
                    text.appendChild(text_text)
                    # beliefs
                    ebeliefs = doc.createElement('beliefs')
                    arg.appendChild(ebeliefs)
                    ebelief = doc.createElement('belief')
                    ebelief.setAttribute('type', str(be_file_record_dict['event'][i]['type']))
                    ebelief.setAttribute('polarity', 'pos')
                    ebelief.setAttribute('sarcasm', 'no')
                    ebeliefs.appendChild(ebelief)
                    if 'source_ere_id' in be_file_record_dict['event'][i]:  # 即非空
                        esource = doc.createElement('source')
                        esource.setAttribute('ere_id', str(be_file_record_dict['event'][i]['source_ere_id']))
                        esource.setAttribute('offset', str(be_file_record_dict['event'][i]['source_offset']))
                        esource.setAttribute('length', str(be_file_record_dict['event'][i]['source_length']))
                        esource_text = doc.createTextNode(be_file_record_dict['event'][i]['source_text'])
                        esource.appendChild(esource_text)
                        ebelief.appendChild(esource)

    # **sentiment**
    st = doc.createElement('sentiment_annotations')
    best.appendChild(st)

    # entity
    if 'entity' in st_file_record_dict:
        entities = doc.createElement('entities')
        st.appendChild(entities)
        for i in range(len(st_file_record_dict['entity'])):
            entity = doc.createElement('entity')
            entity.setAttribute('ere_id', str(st_file_record_dict['entity'][i]['ere_id']))
            entity.setAttribute('offset', str(st_file_record_dict['entity'][i]['offset']))
            entity.setAttribute('length', str(st_file_record_dict['entity'][i]['length']))
            entities.appendChild(entity)

            text = doc.createElement('text')
            text_text = doc.createTextNode(str(st_file_record_dict['entity'][i]['text']))
            text.appendChild(text_text)
            entity.appendChild(text)

            sentiments = doc.createElement('sentiments')
            entity.appendChild(sentiments)

            sentiment = doc.createElement('sentiment')
            sentiment.setAttribute('polarity', str(st_file_record_dict['entity'][i]['polarity']))  # 改下，现在这是列表，改成neg，pos
            sentiment.setAttribute('sarcasm', str(st_file_record_dict['entity'][i]['sarcasm']))
            sentiments.appendChild(sentiment)

            if st_file_record_dict['entity'][i]['polarity'] != 'none':
                if 'source_ere_id' in st_file_record_dict['entity'][i]:  # 即非空
                    source = doc.createElement('source')
                    source.setAttribute('ere_id', str(st_file_record_dict['entity'][i]['source_ere_id']))
                    source.setAttribute('offset', str(st_file_record_dict['entity'][i]['source_offset']))
                    source.setAttribute('length', str(st_file_record_dict['entity'][i]['source_length']))
                    source_text = doc.createTextNode(st_file_record_dict['entity'][i]['source_text'])
                    source.appendChild(source_text)
                    sentiment.appendChild(source)

    # relations
    if 'relation' in st_file_record_dict:
        relations = doc.createElement('relations')
        st.appendChild(relations)
        for i in range(len(st_file_record_dict['relation'])):
            relation = doc.createElement('relation')
            relation.setAttribute('ere_id', str(st_file_record_dict['relation'][i]['ere_id']))
            relations.appendChild(relation)

            if 'trigger_length' in st_file_record_dict['relation'][i]:
                trigger = doc.createElement('trigger')
                trigger_text = doc.createTextNode(str(st_file_record_dict['relation'][i]['trigger_text']))
                trigger.setAttribute('offset', str(int(st_file_record_dict['relation'][i]['trigger_offset'])))
                trigger.setAttribute('length', str(int(st_file_record_dict['relation'][i]['trigger_length'])))
                trigger.appendChild(trigger_text)
                relation.appendChild(trigger)

            rsentiments = doc.createElement('sentiments')
            relation.appendChild(rsentiments)

            rsentiment = doc.createElement('sentiment')
            rsentiment.setAttribute('polarity', str(st_file_record_dict['relation'][i]['polarity']))
            rsentiment.setAttribute('sarcasm', 'no')
            rsentiments.appendChild(rsentiment)

            if st_file_record_dict['relation'][i]['polarity'] != 'none':
                if 'source_ere_id' in st_file_record_dict['relation'][i]:  # 即非空
                    rsource = doc.createElement('source')
                    rsource.setAttribute('ere_id', str(st_file_record_dict['relation'][i]['source_ere_id']))
                    rsource.setAttribute('offset', str(st_file_record_dict['relation'][i]['source_offset']))
                    rsource.setAttribute('length', str(st_file_record_dict['relation'][i]['source_length']))
                    rsource_text = doc.createTextNode(st_file_record_dict['relation'][i]['source_text'])
                    rsource.appendChild(rsource_text)
                    rsentiment.appendChild(rsource)

    # events
    if 'event' in st_file_record_dict:
        events = doc.createElement('events')
        st.appendChild(events)
        for i in range(len(st_file_record_dict['event'])):
            event = doc.createElement('event')
            event.setAttribute('ere_id', str(st_file_record_dict['event'][i]['ere_id']))
            events.appendChild(event)

            etrigger = doc.createElement('trigger')
            etrigger_text = doc.createTextNode(str(st_file_record_dict['event'][i]['trigger_text']))
            etrigger.setAttribute('offset', str(st_file_record_dict['event'][i]['trigger_offset']))
            etrigger.setAttribute('length', str(st_file_record_dict['event'][i]['trigger_length']))
            etrigger.appendChild(etrigger_text)
            event.appendChild(etrigger)

            esentiments = doc.createElement('sentiments')
            event.appendChild(esentiments)

            esentiment = doc.createElement('sentiment')
            esentiment.setAttribute('polarity', str(st_file_record_dict['event'][i]['polarity']))
            esentiment.setAttribute('sarcasm', 'no')
            esentiments.appendChild(esentiment)

            if st_file_record_dict['event'][i]['polarity'] != 'none':
                if 'source_ere_id' in st_file_record_dict['event'][i]:  # 即非空
                    esource = doc.createElement('source')
                    esource.setAttribute('ere_id', str(st_file_record_dict['event'][i]['source_ere_id']))
                    esource.setAttribute('offset', str(st_file_record_dict['event'][i]['source_offset']))
                    esource.setAttribute('length', str(st_file_record_dict['event'][i]['source_length']))
                    esource_text = doc.createTextNode(st_file_record_dict['event'][i]['source_text'])
                    esource.appendChild(esource_text)
                    esentiment.appendChild(esource)

    # 写入文件
    f = open(filepath, 'w')
    # f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))
    f.write(doc.toprettyxml())
    f.close()

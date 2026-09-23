# 医学影像 RAG 评估题目 30 题

说明：这些题目根据公开医学影像网页资料整理并中文改写，用于测试医学影像学 RAG 系统。`expected_evidence_ids` 在 JSON 中留空，因为外部网页题目无法自动对应到你本地索引的 evidence_id；如需跑检索指标，可后续人工补充本地证据 ID。

## 题目与答案

### web_rad_001
题目：疑似急性卒中患者为什么通常首先做头颅 CT 或头颅 MRI？

答案：因为卒中的治疗取决于类型，首先需要区分缺血性卒中和出血性卒中。头颅 CT 能快速发现脑内出血、部分缺血改变和其他急症；头颅 MRI 可更细致评估脑组织损伤。必要时可加做 CTA、CTP、MRA 或 MRP 来评估血管和灌注情况。

来源：https://www.radiologyinfo.org/en/info/stroke

### web_rad_002
题目：急性卒中评估中，CTA 和 CTP 分别主要帮助回答什么问题？

答案：CTA 主要显示脑血管，用于寻找血管狭窄、闭塞、血栓或动脉瘤等异常；CTP 主要评估脑血流灌注，帮助判断缺血核心、低灌注区域和是否适合再通治疗。

来源：https://www.radiologyinfo.org/en/info/stroke

### web_rad_003
题目：头颅 CT 在急诊头痛、头外伤或卒中症状中主要能发现哪些重要问题？

答案：头颅 CT 可快速发现颅内出血、脑损伤、颅骨骨折、动脉瘤破裂相关出血、脑内血栓或出血性卒中表现，也可评估脑肿瘤、脑积水等其他颅内异常。

来源：https://www.radiologyinfo.org/en/info/headct

### web_rad_004
题目：非增强头颅 CT 上急性脑实质出血的典型影像表现是什么？

答案：急性脑实质出血通常表现为脑实质内高密度血肿，周围可有低密度水肿带，并可伴有占位效应、脑室受压、中线移位或破入脑室等表现。非增强 CT 是急诊快速识别急性出血的重要检查。

来源：https://www.radiologyinfo.org/en/info/headct

### web_rad_005
题目：外伤后硬膜外血肿和硬膜下血肿在头颅 CT 形态上如何区分？

答案：硬膜外血肿多呈双凸透镜形或梭形颅板下高密度影，常受颅缝限制，常伴颅骨骨折；硬膜下血肿多呈新月形沿脑表面分布，可跨越颅缝，但通常受大脑镰、天幕等硬膜反折限制。

来源：https://www.radiologyinfo.org/en/info/headct

### web_rad_006
题目：蛛网膜下腔出血疑诊时，头颅 CT 和 CTA 的作用分别是什么？

答案：头颅 CT 用于快速发现蛛网膜下腔内出血，尤其是突发剧烈头痛患者；CTA 可进一步评估脑血管，帮助寻找破裂动脉瘤或其他血管异常。

来源：https://www.radiologyinfo.org/en/info/headct

### web_rad_007
题目：胸部 X 线在临床上常用于评估哪些胸部问题？

答案：胸部 X 线常用于评估呼吸困难、持续咳嗽、发热、胸痛或外伤，也可帮助诊断或随访肺炎、心力衰竭、肺气肿、肺癌、医疗器械位置以及胸腔积液或气胸等。

来源：https://www.radiologyinfo.org/en/info/chestrad

### web_rad_008
题目：胸部 X 线上骨、软组织和含气肺组织通常分别呈什么密度表现？

答案：骨组织吸收 X 线较多，通常呈白色或浅灰色；软组织呈不同灰度；含气肺组织吸收较少，通常呈较黑的透亮影。

来源：https://www.radiologyinfo.org/en/info/chestrad

### web_rad_009
题目：为什么正常或不典型的胸片不能排除肺栓塞？

答案：胸片对肺栓塞本身不敏感，许多肺栓塞患者胸片可正常或仅有非特异表现。胸片主要用于排除肺炎、气胸、心衰等其他胸痛或呼吸困难原因；疑似肺栓塞时通常需要根据风险分层进一步做 D-二聚体、CTPA 或 V/Q 扫描等。

来源：https://www.merckmanuals.com/professional/pulmonary-disorders/pulmonary-embolism-pe/pulmonary-embolism-pe

### web_rad_010
题目：气胸在直立胸片上的典型影像表现有哪些？

答案：典型表现包括胸膜腔内透亮气体影、可见脏层胸膜线、该线外周肺纹理消失，肺组织向肺门或内侧萎陷。大量或张力性气胸还可出现纵隔或气管向对侧移位。

来源：https://www.merckmanuals.com/professional/pulmonary-disorders/mediastinal-and-pleural-disorders/pneumothorax

### web_rad_011
题目：张力性气胸为什么不能等待影像学确认后再处理？

答案：张力性气胸会使胸膜腔压力持续升高，导致肺萎陷、纵隔移位和静脉回流受阻，可能迅速出现低血压、呼吸循环衰竭或心脏骤停。因此一旦临床高度怀疑，应立即减压处理，不能因等待胸片或 CT 而延误。

来源：https://www.merckmanuals.com/professional/pulmonary-disorders/mediastinal-and-pleural-disorders/pneumothorax

### web_rad_012
题目：仰卧位胸片中提示气胸的 deep sulcus sign 指什么？

答案：仰卧位时胸膜腔气体可聚集在前下胸腔，不一定表现为肺尖气胸。deep sulcus sign 指患侧肋膈角异常加深、透亮，提示仰卧位气胸，尤其适用于创伤或不能站立患者。

来源：https://www.merckmanuals.com/professional/pulmonary-disorders/mediastinal-and-pleural-disorders/pneumothorax

### web_rad_013
题目：疑似急性肺栓塞时，CT 肺动脉造影和 V/Q 扫描各在什么情况下更常用？

答案：CT 肺动脉造影通常是诊断急性肺栓塞的首选影像检查，可直接显示肺动脉内充盈缺损并评估其他肺部病变。若存在肾功能不全、碘对比剂禁忌或妊娠等情况，且胸片正常或接近正常，V/Q 扫描常作为替代选择。

来源：https://www.merckmanuals.com/professional/pulmonary-disorders/pulmonary-embolism-pe/pulmonary-embolism-pe

### web_rad_014
题目：肺栓塞在胸片上有哪些经典但不敏感的提示性表现？

答案：胸片可正常或仅有非特异表现。经典但不敏感的提示性表现包括局灶血管纹理减少的 Westermark sign、胸膜基底楔形阴影即 Hampton hump、右下肺动脉增宽、局灶浸润、肺不张、膈肌升高或少量胸腔积液等。

来源：https://www.merckmanuals.com/professional/pulmonary-disorders/pulmonary-embolism-pe/pulmonary-embolism-pe

### web_rad_015
题目：与胸片相比，胸部 CT 在胸部疾病评估中有哪些优势？

答案：胸部 CT 可提供横断面和多平面图像，对肺结节、肿瘤、肺炎、结核、支气管扩张、胸膜疾病、间质性肺病、胸部外伤和血管病变等显示更详细；必要时还可做 CTA 评价胸部血管。

来源：https://www.radiologyinfo.org/en/info/chestct

### web_rad_016
题目：胸片发现肺炎可疑但临床或影像不典型时，为什么可能需要进一步做胸部 CT？

答案：胸片是肺炎等胸部疾病常用初筛检查，但对小病灶、复杂病变、并发症或被重叠结构遮挡的异常有限。胸部 CT 能更清楚显示肺实质、胸膜和纵隔结构，有助于评估肺炎范围、并发症以及其他可能诊断。

来源：https://www.radiologyinfo.org/en/info/chestct

### web_rad_017
题目：腹部和盆腔 CT 常用于评估哪些腹盆腔疾病或急症？

答案：腹部和盆腔 CT 常用于寻找腹痛、恶心呕吐原因，评估阑尾炎、憩室炎、肾盂肾炎、感染性积液或脓肿、炎症性肠病、胰腺炎、肝肠肾胰等肿瘤、肾和膀胱结石、腹主动脉瘤以及外伤导致的实质脏器损伤和出血。

来源：https://www.radiologyinfo.org/en/info/abdominct

### web_rad_018
题目：成人疑似急性阑尾炎时，哪种影像检查常被优先用于诊断，理由是什么？

答案：成人疑似急性阑尾炎时，腹部和盆腔 CT 常被优先用于诊断，因为它准确性高，能显示阑尾炎本身及周围炎症、穿孔、脓肿等并发症，也能帮助排除其他腹痛原因。

来源：https://www.radiologyinfo.org/en/info/appendicitis

### web_rad_019
题目：儿童或孕妇疑似阑尾炎时，为什么常先考虑超声或 MRI？

答案：儿童和孕妇需要尽量减少电离辐射暴露。超声不使用电离辐射，常作为儿童和孕妇阑尾炎评估的优先检查；若超声结果不明确，孕妇或年轻患者可考虑 MRI 进一步评估。

来源：https://www.radiologyinfo.org/en/info/appendicitis

### web_rad_020
题目：阑尾炎破裂形成腹腔脓肿时，影像引导经皮脓肿引流的作用是什么？

答案：当阑尾炎破裂形成脓肿时，经皮脓肿引流可在影像引导下将感染性液体从体内引出，控制感染和脓肿负荷，常与后续或同步的外科处理、抗感染治疗等共同使用。

来源：https://www.radiologyinfo.org/en/info/appendicitis

### web_rad_021
题目：急性胆囊炎的腹部超声常见诊断线索有哪些？

答案：常见线索包括胆囊结石或胆囊颈/胆囊管梗阻证据、胆囊壁增厚、胆囊周围液体、超声 Murphy 征阳性，以及胆囊扩张等。若超声结果不明确，可考虑胆道核素显像；怀疑穿孔、脓肿或胆总管结石时可进一步用 CT、MRI/MRCP。

来源：https://www.merckmanuals.com/professional/hepatic-and-biliary-disorders/gallbladder-and-bile-duct-disorders/acute-cholecystitis

### web_rad_022
题目：无结石性胆囊炎在影像上应如何怀疑？

答案：高危患者如重症、长期禁食、免疫抑制或休克患者，即使没有胆结石，如果出现超声 Murphy 征、胆囊壁增厚、胆囊周围液体、胆囊扩张或胆泥，也应怀疑无结石性胆囊炎。胆道核素显像对该病较可靠，CT/MRI 多作为补充。

来源：https://www.merckmanuals.com/professional/hepatic-and-biliary-disorders/gallbladder-and-bile-duct-disorders/acute-cholecystitis

### web_rad_023
题目：急性胰腺炎中，超声、CT 和 MRCP 各自主要用于解决什么问题？

答案：超声主要用于寻找胆囊结石和胆道梗阻等病因；腹部 CT 可显示胰腺和周围结构，评估炎症、梗阻及并发症；MRCP 能详细显示肝胆胰管系统，对胆道树内结石或导管阻塞评估有价值。

来源：https://www.radiologyinfo.org/en/info/pancreatitis

### web_rad_024
题目：增强横断面影像在坏死性胰腺炎评估中的价值是什么？

答案：增强 CT 或其他增强横断面影像可显示胰腺或胰周组织坏死、液体积聚、包裹性坏死、感染征象和其他局部并发症。坏死性胰腺炎通常提示病程更重、住院时间更长且并发症风险更高。

来源：https://www.merckmanuals.com/professional/gastrointestinal-disorders/pancreatitis/acute-pancreatitis

### web_rad_025
题目：MRCP 在胆源性胰腺炎或胆道梗阻评估中有什么优势？

答案：MRCP 是 MRI 技术，可清楚显示肝脏、胆囊、胆管、胰腺和胰管，适合评估胆总管或胆道树内结石、胆道梗阻和胰胆管解剖。某些胆道结石在超声或 CT 上不容易显示，MRCP 可提供补充信息。

来源：https://www.radiologyinfo.org/en/info/pancreatitis

### web_rad_026
题目：腹部 CT 在泌尿系结石和腹主动脉瘤评估中有什么作用？

答案：腹部和盆腔 CT 可用于寻找肾和膀胱结石，判断结石部位、梗阻和相关并发症；也可评估腹主动脉瘤，帮助判断动脉瘤存在、范围以及是否有急性并发症或其他腹痛原因。

来源：https://www.radiologyinfo.org/en/info/abdominct

### web_rad_027
题目：筛查性乳腺 X 线摄影的主要目的是什么？

答案：筛查性乳腺 X 线摄影使用低剂量 X 线在无症状阶段尽早发现乳腺癌或癌前病变，因为早期发现时治疗选择更多、预后更好。它也能发现一些尚不能触及的肿块、密度异常或钙化。

来源：https://www.radiologyinfo.org/en/info/mammo

### web_rad_028
题目：乳腺断层合成成像，也称 3D mammography，相比常规二维乳腺摄影可能有哪些优势？

答案：乳腺断层合成成像从多个角度采集图像并重建为三维图像组，可减少组织重叠影响，更容易发现被常规图像遮盖的小癌灶，减少不必要召回，并更准确判断病灶大小、形态和位置，尤其对致密乳腺更有帮助。

来源：https://www.radiologyinfo.org/en/info/mammo

### web_rad_029
题目：乳腺肿块评估中，乳腺超声和乳腺 MRI 各自常用于补充哪些信息？

答案：乳腺超声可帮助判断肿块是实性还是囊性，并能评估乳腺摄影较难显示的区域；乳腺 MRI 可提供乳腺内部更详细的软组织信息，常用于乳腺摄影或超声不易显示的病灶、致密乳腺或需要评估病变范围的情况。

来源：https://www.radiologyinfo.org/en/info/breast-cancer

### web_rad_030
题目：BI-RADS 3、4、5 类在乳腺影像报告中分别大致提示什么管理策略？

答案：BI-RADS 3 表示大概率良性，但仍建议短期随访，常见为 6 个月复查；BI-RADS 4 表示可疑异常，通常建议活检，并可细分 4A、4B、4C；BI-RADS 5 表示高度提示恶性，强烈建议活检以确诊。

来源：https://www.radiologyinfo.org/en/info/article-breast-imaging-report

## 主要来源

- RadiologyInfo, Stroke: https://www.radiologyinfo.org/en/info/stroke
- RadiologyInfo, Head CT: https://www.radiologyinfo.org/en/info/headct
- RadiologyInfo, Chest X-ray: https://www.radiologyinfo.org/en/info/chestrad
- RadiologyInfo, Chest CT: https://www.radiologyinfo.org/en/info/chestct
- Merck Manual Professional, Pneumothorax: https://www.merckmanuals.com/professional/pulmonary-disorders/mediastinal-and-pleural-disorders/pneumothorax
- Merck Manual Professional, Pulmonary Embolism: https://www.merckmanuals.com/professional/pulmonary-disorders/pulmonary-embolism-pe/pulmonary-embolism-pe
- RadiologyInfo, Abdominal and Pelvic CT: https://www.radiologyinfo.org/en/info/abdominct
- RadiologyInfo, Appendicitis: https://www.radiologyinfo.org/en/info/appendicitis
- Merck Manual Professional, Acute Cholecystitis: https://www.merckmanuals.com/professional/hepatic-and-biliary-disorders/gallbladder-and-bile-duct-disorders/acute-cholecystitis
- RadiologyInfo, Pancreatitis: https://www.radiologyinfo.org/en/info/pancreatitis
- Merck Manual Professional, Acute Pancreatitis: https://www.merckmanuals.com/professional/gastrointestinal-disorders/pancreatitis/acute-pancreatitis
- RadiologyInfo, Mammography: https://www.radiologyinfo.org/en/info/mammo
- RadiologyInfo, Breast Cancer: https://www.radiologyinfo.org/en/info/breast-cancer
- RadiologyInfo, BI-RADS report guide: https://www.radiologyinfo.org/en/info/article-breast-imaging-report

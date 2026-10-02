**دليل التعامل مع الزنك والمعادن في تحليل التفاعلات**  
**Zinc & Metalloprotein Interaction Tutorial**  
***الأداة:*** * * *gmx_2d_interaction_diagram.py* * (الإصدار v38c فأحدث)*  
 *  
 * ***الغرض:*** * كل ما يخص كشف وإظهار تنسيق المعادن (الزنك خاصةً) في مخطط التفاعلات ثنائي الأبعاد.*  
 *  
 * ***قاعدة ذهبية:*** * كل الضبط يتم من * ***سطر الأوامر (الترمينال)*** * فقط — لا تعديل على الكود الأصلي إطلاقاً.*  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANElEQVR4nO3OMQ0AIAwAwZIgBKn1gjJsdGLBABMhuZt+/JaZIyJmAADwi9VP1NMNAABu1AaU4gUeBSGW2wAAAABJRU5ErkJggg==)  
**٠. المبدأ العلمي أولاً (اقرأه قبل أي شيء)**  
تنسيق المعدن الحقيقي (Metal coordination) له خصائص فيزيائية ثابتة:  
| | |  
|-|-|  
| **الخاصية** | **القيمة الحقيقية للزنك** |   
| مسافة التنسيق المثالية Zn–N/O/S | **2.0 – 2.4 Å** |   
| الحد الأقصى المقبول (مع حركة MD) | **~3.2 Å** |   
| ذرات التنسيق الشائعة | N (هيستيدين)، O (أسبارتات/جلوتامات)، S (سيستئين) |   
| نسبة الإشغال لتنسيق مستقر | **30% – 90%** من الإطارات |   
   
**القاعدة الحاسمة:**  
- تنسيق موجود في **>30%** من الإطارات = تناسق حقيقي مستقر ✓  
- تنسيق في **5–30%** = تناسق ضعيف/متذبذب (اذكره بحذر)  
- تنسيق في **<5%** = تلامس عابر بالصدفة،  **ليس تنسيقاً** ✗  
- مسافة **>3.5 Å** = ليست تنسيقاً معدنياً، مجرد قرب مكاني ✗  
*⚠️ * ***تحذير أخلاقي:*** * لا ترفع * *--min-metal* * أو * *--metal-cutoff* * لأرقام مبالغ فيها لمجرد "إظهار" الزنك. إن ظهر برقم مزيّف وأعاد مُحكّم المحاكاة، ستفقد مصداقيتك. أظهر الحقيقة فقط.*  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANklEQVR4nO3OQQmAABRAsScYxpg/h5VMYARvRrCCNxG2BFtmZquOAAD4i3Ot7mr/egIAwGvXA224BcUMk6pDAAAAAElFTkSuQmCC)  
**١. المعاملات الأربعة الخاصة بالمعادن**  
كل الضبط يتم عبر هذه المعاملات في الترمينال:  
| | | |  
|-|-|-|  
| **المعامل** | **الافتراضي** | **ماذا يفعل** |   
| --metal-selection | "resname ZN ZN2 ZNB MG MG2 MN FE FE2 CU NI CO CA" | أسماء ذرات المعادن التي يبحث عنها |   
| --metal-cutoff | 3.0 | أقصى مسافة (Å) تُعتبر تنسيقاً |   
| --min-metal | 5.0 | أقل نسبة مئوية (%) لإظهار التنسيق في الصورة |   
| --min-percent | 4.0 | العتبة العامة لباقي التفاعلات (لا تخص المعدن) |   
   
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OMQ2AABAAsSPBCUZfE2IYmVDBhAU2QtIq6DIzW7UHAMBfnGt1V8fXEwAAXrse/xcF7U7sx4wAAAAASUVORK5CYII=)  
**٢. سير العمل التشخيصي (اتبعه بالترتيب)**  
**الخطوة أ — اعرف اسم المعدن الفعلي في نظامك**  
echo q | gmx make_ndx -f MD.tpr -o /dev/null 2>&1 | grep -iE "zn|mg|mn|fe|cu|ni|co|zinc|ion"  
   
**ما ستراه (مثال حقيقي من نظامك):**  
13 ZN2                 :     1 atoms  
   
← اسم الزنك عندك هو ZN2، وهو موجود في --metal-selection الافتراضي. ممتاز.  
**إذا ظهر اسم غريب** (مثل Zn بحرف صغير أو ZN1):  
   
 الكود v38c يكتشفه تلقائياً (مطابقة غير حساسة لحالة الأحرف)، لكن يمكنك تحديده صراحةً:  
--metal-selection "resname ZN2"  
   
**الخطوة ب — شغّل بالإعدادات الافتراضية أولاً**  
python3 gmx_2d_interaction_diagram.py \  
   --topol MD.tpr --traj MD_center.xtc \  
   --ligand-selection "resname LIG0" --protein-selection "protein" \  
   --metal-selection "resname ZN ZN2 ZNB MG MG2 MN FE FE2 CU NI CO CA" \  
   --ligand-ref LIG.mol2 --output-prefix diagram \  
   --full-report --binding-rmsf --analyze-all \  
   --metal-cutoff 3.0 \  
   --start 0 --stop 1000 --stride 1 --prolif-jobs 4 --dpi 600  
   
**الخطوة ج — اقرأ رسائل التشخيص في الترمينال**  
الكود يطبع تشخيصاً كاملاً. ابحث عن هذه السطور:  
[info] Metal atoms  : 1  resnames={'ZN2'}  names={'ZN'}  
 [info] Metal custom occurrences: 1  
 [info] Closest metal approach: ZN2378 … ligand N(N6) = 3.06 Å (cutoff=3.0 Å)  
 [metal] ZN2 coordination present in 0.1% of frames (threshold --min-metal=5.0%)  
   
**كيف تفسّرها:**  
| | |  
|-|-|  
| **السطر** | **المعنى** |   
| Metal atoms : 1 resnames={'ZN2'} | الزنك اكتُشف بنجاح (الاسم صحيح) |   
| Closest metal approach: … = 3.06 Å | أقرب مسافة وصلها الزنك من الليجند |   
| coordination present in 0.1% | **النسبة الحقيقية للتنسيق** ← الرقم الأهم |   
   
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OMQ2AUBBAsUeCE4yeIiT9CRVMWGAjJK2CbjNzVGcAAPzF2qu7Wl9PAAB47XoA/vcF8exqpY4AAAAASUVORK5CYII=)  
**٣. جدول القرار: ماذا تكتب حسب ما يظهر**  
**الحالة ١: الزنك لا يُكتشف أصلاً**  
[warning] --metal-selection matched 0 atoms  
 [hint] Metal-like residue names actually present: ['ZN2']  
 [hint] Re-run with --metal-selection "resname ZN2"  
   
**الحل:** انسخ الاسم من الـ hint:  
--metal-selection "resname ZN2"  
   
**الحالة ٢: الزنك قريب لكن أبعد قليلاً من cutoff**  
[info] Metal custom occurrences: 0  
 [info] Closest metal approach: ZN2378 … = 3.06 Å (cutoff=3.0 Å)  
 [note] re-run with --metal-cutoff 3.2  
   
**الحل:** استخدم القيمة المقترحة بالضبط (لا تبالغ):  
--metal-cutoff 3.2  
   
**الحالة ٣: التنسيق موجود لكن نسبته أقل من العتبة**  
[metal] ZN2 coordination present in 0.1% of frames (threshold --min-metal=5.0%)  
 [metal][note] This is BELOW the threshold and will be hidden.  
               To show it, re-run with --min-metal 1.0  
   
**القرار العلمي مهم هنا:**  
- إذا النسبة **≥5%** → سيظهر تلقائياً، لا تفعل شيئاً  
- إذا النسبة **1–5%** → تناسق ضعيف. أظهره فقط إن أردت، مع ذكر النسبة في بحثك:  
- --min-metal 1.0  
   
- إذا النسبة **<1%** (مثل 0.1%) →  **تلامس عابر، ليس تنسيقاً**. الأصح  **عدم إظهاره**. إن أصررت للتوثيق فقط:  
- --min-metal 0.05  
   
-   
 لكن **يجب** أن تكتب في بحثك "تلامس عابر في 0.1% من الإطارات".  
**الحالة ٤: التنسيق قوي وحقيقي**  
[metal] ZN2 coordination present in 45.0% of frames (threshold --min-metal=5.0%)  
   
**ممتاز!** تناسق حقيقي مستقر. يظهر تلقائياً بلا أي تعديل. هذا ما تريده للنشر.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAAM0lEQVR4nO3OMQ0AIAwAwZKQ6kBqjSAOJywYYCIkd9OP36pqRMQMAAB+sfqJfLoBAMCN3NYoAzBA+QG0AAAAAElFTkSuQmCC)  
**٤. الأوامر الجاهزة لكل حالة (انسخ والصق)**  
**أ) التشغيل القياسي (بروتين معدني، إعدادات علمية صحيحة)**  
python3 gmx_2d_interaction_diagram.py \  
   --topol MD.tpr --traj MD_center.xtc \  
   --ligand-selection "resname LIG0" --protein-selection "protein" \  
   --metal-selection "resname ZN ZN2 ZNB MG MG2 MN FE FE2 CU NI CO CA" \  
   --ligand-ref LIG.mol2 --output-prefix diagram \  
   --full-report --binding-rmsf --analyze-all \  
   --metal-cutoff 3.2 \  
   --start 0 --stop 1000 --stride 1 --prolif-jobs 4 --dpi 600  
   
**ب) الزنك أبعد قليلاً — توسيع المسافة (بحد أقصى 3.5)**  
python3 gmx_2d_interaction_diagram.py \  
   --topol MD.tpr --traj MD_center.xtc \  
   --ligand-selection "resname LIG0" --protein-selection "protein" \  
   --metal-selection "resname ZN ZN2 ZNB MG MG2 MN FE FE2 CU NI CO CA" \  
   --ligand-ref LIG.mol2 --output-prefix diagram \  
   --full-report --binding-rmsf --analyze-all \  
   --metal-cutoff 3.5 --min-metal 1.0 \  
   --start 0 --stop 1000 --stride 1 --prolif-jobs 4 --dpi 600  
   
**ج) إظهار تنسيق ضعيف للتوثيق (مع الأمانة العلمية)**  
python3 gmx_2d_interaction_diagram.py \  
   --topol MD.tpr --traj MD_center.xtc \  
   --ligand-selection "resname LIG0" --protein-selection "protein" \  
   --metal-selection "resname ZN ZN2 ZNB MG MG2 MN FE FE2 CU NI CO CA" \  
   --ligand-ref LIG.mol2 --output-prefix diagram \  
   --full-report --binding-rmsf --analyze-all \  
   --metal-cutoff 3.2 --min-metal 0.05 \  
   --start 0 --stop 1000 --stride 1 --prolif-jobs 4 --dpi 600  
   
*⚠️ استخدم هذا فقط للاستكشاف. للنشر، اذكر النسبة الحقيقية صراحةً.*  
**د) اسم زنك مخصص (إذا كان غير قياسي)**  
python3 gmx_2d_interaction_diagram.py \  
   --topol MD.tpr --traj MD_center.xtc \  
   --ligand-selection "resname LIG0" --protein-selection "protein" \  
   --metal-selection "resname ZN2" \  
   --ligand-ref LIG.mol2 --output-prefix diagram \  
   --full-report --binding-rmsf --analyze-all \  
   --metal-cutoff 3.2 \  
   --start 0 --stop 1000 --stride 1 --prolif-jobs 4 --dpi 600  
   
**هـ) تشغيل سريع للاختبار فقط (إطارات قليلة)**  
python3 gmx_2d_interaction_diagram.py \  
   --topol MD.tpr --traj MD_center.xtc \  
   --ligand-selection "resname LIG0" --protein-selection "protein" \  
   --metal-selection "resname ZN ZN2 ZNB MG MG2 MN FE FE2 CU NI CO CA" \  
   --ligand-ref LIG.mol2 --output-prefix diagram_test \  
   --metal-cutoff 3.2 \  
   --start 0 --stop 100 --stride 20  
   
*بدون * *--full-report* * وبإطارات قليلة — فقط لرؤية رسائل تشخيص الزنك بسرعة.*  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANklEQVR4nO3OQQmAABRAsScYxpg/h5VMYARvRrCCNxG2BFtmZquOAAD4i3Ot7mr/egIAwGvXA224BcUMk6pDAAAAAElFTkSuQmCC)  
**٥. معاملات المجموعات (Groups) لنظام الزنك**  
في نظام zinc metalloprotein، الزنك يأخذ رقم مجموعة (عادة 13)، فيزيح الليجند لرقم أعلى.  
| | | |  
|-|-|-|  
| **التحليل** | **المجموعة** | **المعامل** |   
| Protein RMSD | 3 (C-alpha) | --rmsd-protein-g1 3 --rmsd-protein-g2 3 |   
| Ligand RMSD | fit=3, LIG=14 | --rmsd-ligand-g1 3 --rmsd-ligand-g2 14 |   
| Protein RMSF | 4 (Backbone) | --rmsf-protein-group 4 |   
| Ligand RMSF | 14 | --rmsf-ligand-group 14 |   
| H-bond | Protein=1, LIG=14 | --hbond-g1 1 --hbond-g2 14 |   
| Gyrate | 1 (Protein) | --gyrate-group 1 |   
| SASA | Protein=1, LIG=14 | --sasa-protein-group 1 --sasa-ligand-group 14 |   
   
***ملاحظة:*** * الإصدار v37+ يستخدم الليجند=14 افتراضياً (لأن الزنك أخذ 13). تحقق من أرقام مجموعاتك:*  
*echo q | gmx make_ndx -f MD.tpr -o /dev/null 2>&1 | grep -E "^ *[0-9]+ "  
 *  
*إذا كان ترتيب مجموعاتك مختلفاً، مرّر الأرقام الصحيحة يدوياً في الأمر.*  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OMQ2AUBBAsUfyRTCh9VRgEBGsWGAjJK2CbjNzVGcAAPzFtapV7V9PAAB47X4AEWgEMAY9+pUAAAAASUVORK5CYII=)  
**٦. كيف يُكتشف تنسيق الزنك داخلياً (للفهم)**  
الكود يقيس المسافة بين **ذرة الزنك** وبين  **ذرات الليجند القادرة على التنسيق** (N, O, S فقط — تُسمّى COORD_EL). في كل إطار:  
1. يحسب المسافة Zn ↔ كل ذرة N/O/S في الليجند  
2. إذا المسافة ≤ --metal-cutoff → يسجّل تنسيقاً في ذلك الإطار  
3. في النهاية: النسبة = (عدد الإطارات ذات التنسيق ÷ كل الإطارات) × 100  
4. إذا النسبة ≥ --min-metal → يُرسم في الصورة  
**نقطة مهمة:** الكود يقيس فقط ذرات  **N/O/S**. إذا كان الزنك يتناسق مع ذرة أخرى في ليجندك، قد لا يُحسب. تأكد أن ذرة التنسيق المتوقعة هي N أو O أو S.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OQQmAABRAsSd49m4v6wg/pwmMYQVvImwJtszMXp0BAPAX91pt1fH1BACA164Hoq8EQMMPmF8AAAAASUVORK5CYII=)  
**٧. قائمة تحقق سريعة (Checklist)**  
قبل أن تقول "الزنك لا يظهر"، تحقق بالترتيب:  
- **الاسم:** هل ظهر Metal atoms : 1 resnames={...}؟ إن لا → صحّح --metal-selection  
- **المسافة:** ما قيمة Closest metal approach = X.XX Å؟ إن > cutoff → ارفع --metal-cutoff للقيمة المقترحة  
- **النسبة:** ما قيمة coordination present in XX%؟  
  - ≥5% → يظهر تلقائياً  
  - 1–5% → --min-metal 1.0  
  - <1% → تلامس عابر، الأصح عدم إظهاره  
- **ذرة التنسيق:** هل ذرة الليجند المتوقعة N/O/S؟ (الكود لا يقيس الكربون)  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANklEQVR4nO3OMQ2AABAAsSNBCkJfFSqwwIgHRiywEZJWQZeZ2ao9AAD+4lyruzq+ngAA8Nr1AOH8BeZxN/IIAAAAAElFTkSuQmCC)  
**٨. ملخص القيم الموصى بها (للنشر العلمي)**  
| | | |  
|-|-|-|  
| **المعامل** | **القيمة الآمنة علمياً** | **لا تتجاوز** |   
| --metal-cutoff | 3.0 – 3.2 | 3.5 |   
| --min-metal | 5.0 (افتراضي) | اخفضه فقط بشفافية |   
| --stride | 1 (للزنك النادر) | — |   
   
***المبدأ:*** * أظهر ما تقيسه المحاكاة بصدق. إن كان الزنك لا يتناسق فعلياً (0.1%)، فهذه نتيجة علمية بحد ذاتها — ربما الليجند لا يصل للزنك في وضعه الحالي، أو الـ docking pose يحتاج مراجعة.*  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OYQ1AABSAwY9JoICqL4Z8Ikiggn9mu0twy8wc1RkAAH9xbdVa7V9PAAB47X4A9CgEJQFjJ/EAAAAASUVORK5CYII=)  
*انتهى الدليل. ارجع إليه كلما تعاملت مع بروتين معدني*  
*هذا الكود المسمى    gmx_2d_interaction_diagram_with_binding_site_FINAL.py*  
*يتم تنفيذه من خلال    python3 gmx_2d_interaction_diagram_with_binding_site_FINAL.py \*  
*  --topol MD.tpr \*  
*  --traj MD_center.xtc \*  
*  --ligand-selection "resname LIG" \*  
*  --protein-selection "protein" \*  
*  --metal-selection "resname ZN ZN2 ZNB MG MG2 MN FE FE2 CU NI CO CA" \*  
*  --ligand-ref LIG.mol2 \*  
*  --output-prefix diagram \*  
*  --index index.ndx \*  
*  --full-report \*  
*  --binding-rmsf \*  
*  --analyze-all \*  
*  --metal-cutoff 3.3 \*  
*  --min-metal-frames 10 \*  
*  --start 0 \*  
*  --stop 1000 \*  
*  --stride 5 \*  
*  --prolif-jobs 4 \*  
*  --dpi 600*  
*وهو كود يتم ادخال المتطلبات فيه بصوره يدويه اي انه *  
*يصلح لي الزنك وغير الزنك ويصلح للبايندنج بوكيت وغير البايندنج بوكيت وهو ممتاز*  
   
*هذا بالنسبه للامر السابق *  
*اما امر البي سي اي انالايزر بايثون ففيه امران هما *  
*python3 PCA_FEL_ANALYZER.py single*  
*python3 PCA_FEL_ANALYZER.py compare*  
*اما بما يخص الامر الاخير فهو*  
*python3 pca_fel_pro_for_binding_pocket_in_Zn_model_.py \*  
*  --tpr MD.tpr \*  
*  --xtc MD_center.xtc \*  
*  --index index.ndx*  
*هذا الامر يمكن من خلاله تعديل اسم الاندكس واسم التي بي ار واسم الاكس تي سي بما تريده انت*  

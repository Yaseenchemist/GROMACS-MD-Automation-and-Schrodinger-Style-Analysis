# تشغيل المحاكاة على السيرفر (Aphroditi) — خطوة بخطوة

> الافتراض: أنت الآن **على اللاب توب** داخل المجلد اللي يحتوي:
> `complex.pdb` · `ions.mdp` `EM.mdp` `NVT.mdp` `NPT.mdp` `MD.mdp` · `run_server.sh` · `pull_results.sh`

---

## الخطوة 0 — (على اللاب توب) ارفع المجلد للسيرفر

```bash
# داخل مجلد المشروع على اللاب توب
rsync -avP ./ YOUR_USERNAME@HPC_LOGIN_NODE:/path/to/your_project/gromacs_run/
```
> عدّل `/path/to/your_project/gromacs_run/` لمسار منطقة مشروعك على السيرفر. لو المسار غير متأكد منه، سجّل دخول أولاً (الخطوة 1) واعمل المجلد بـ `mkdir -p` ثم ارفع.

---

## الخطوة 1 — سجّل دخول للـ front node وادخل المجلد

```bash
ssh YOUR_USERNAME@HPC_LOGIN_NODE
cd /path/to/your_project/gromacs_run
ls    # تأكد: complex.pdb + الـ 5 mdp + run_server.sh
```

---

## الخطوة 2 — فحوصات ما قبل التشغيل (مهمة جداً)

```bash
# 1) اسم موديول GROMACS الصحيح
module avail 2>&1 | grep -i gromacs

# 2) أسماء الـ partitions و الـ GPU المتاحة
sinfo -s

# 3) الحساب/partitions المسموحة لك
sacctmgr show assoc user=YOUR_USERNAME format=Account,Partition%20 -p

# 4) هل obabel متوفر؟ (لازم لتحضير الليكند + SwissParam)
command -v obabel || echo "obabel مو موجود — شوف الخطوة 2-b"
```

**2-b — إذا `obabel` مو موجود:** فعّل بيئة conda عندك اللي فيها OpenBabel، أو حمّل موديول:
```bash
module load OpenBabel 2>/dev/null || conda activate <بيئة_فيها_openbabel>
```
> بديل مضمون: حضّر الليكند على اللاب توب (شغّل `run_local.sh` لين ما يتولّد `LIG.itp`، أو أول جزء منه)، وارفع `LIG.pdb` و`LIG.itp` (و`LIG.par` إن وُجد) مع `complex.pdb`. السكربت راح **يتخطّى obabel و SwissParam** تلقائياً.

---

## الخطوة 3 — راجع إعدادات SLURM أعلى `run_server.sh`

```bash
nano run_server.sh
```
طابِق هذي المتغيّرات مع مخرجات الخطوة 2:
```
GMX_MODULE="GROMACS/2025.3-foss-2025a-CUDA-12.9.1"   # من module avail
SLURM_PARTITION="gpu"                                 # من sinfo
SLURM_GRES="gpu:1"                                    # أو gpu:v100:1
SLURM_ACCOUNT="YOUR_PROJECT_ID"
SLURM_CPUS=16
SLURM_TIME="24:00:00"
```
احفظ واخرج (`Ctrl+O`, `Enter`, `Ctrl+X`).

---

## الخطوة 3.5 — فحص وتنظيف `complex.pdb` (إلزامي — يمنع فشل `pdb2gmx`)

> **ليش هذي الخطوة موجودة:** لو في أي residue بروتينية ناقصة ذرات الـ backbone (خصوصاً `CA`) — مثل terminus مكسور، أو residue ناقصة من الـ crystal structure — راح يطيح `pdb2gmx` بخطأ قاتل:
> `Residue ... named XXX ... the atom CA ... is not found in the input file`.
> السبب **مو بالسكربت**، بل بملف `complex.pdb` نفسه. هذي الخطوة تكشف وتشيل أي residue مكسورة **قبل** التشغيل، حتى ما نواجه نفس الخطأ.
> (هذا بالضبط الي صار سابقاً مع `LYS 224` — كانت ناقصة كل الـ backbone وعدها بس `N H1 H2`.)

**0) تأكد إنك بالمجلد الصح وإن `complex.pdb` فيه بروتين + ليكند (قبل أي شي):**

> ⚠️ **مهم جداً — هذا الي سبب أول لخبطة:** الملفات مثل `REC.pdb` · `LIG1.pdb` · `LIG.pdb` · `topol.top` · `conf.gro` · `md.log` **ما تنوجد إلا بعد** ما تشغّل السكربت — **السكربت هو الي يولّدها**. فلو دخلت المجلد ولقيت بس `complex.pdb` + الـ mdp، هذا **طبيعي** ومعناه السكربت لسه ما اشتغل (مو خطأ). ومن أكثر الأخطاء شيوعاً إنك تكون بمجلد غلط (مثلاً `~/` بدل مسار المشروع) فتطلعلك `No such file / cannot open`.

```bash
# 1) تأكد إنك بالمجلد الصح — لازم تشوف complex.pdb + الـ 5 mdp
cd /path/to/your_project/gromacs_run
pwd
ls complex.pdb ions.mdp EM.mdp NVT.mdp NPT.mdp MD.mdp 2>&1

# 2) تأكد complex.pdb فيه بروتين (ATOM) وليكند (HETATM)
echo "عدد ذرات البروتين (ATOM):   $(grep -c '^ATOM'   complex.pdb)"
echo "عدد ذرات الليكند (HETATM):  $(grep -c '^HETATM' complex.pdb)"
```
> لازم **الاثنين أكبر من صفر**:
> - لو `HETATM = 0` → السكربت راح يطيح فوراً بـ `ERROR: ماكو HETATM (ليكند) داخل complex.pdb` (ما راح يلگي ليكند يحضّره بـ SwissParam). صلّح الملف: تأكد إن سطور الليكند مكتوبة `HETATM` مو `ATOM`.
> - لو `ATOM = 0` → الملف غلط أو فارغ أو مو بمكانه. صلّحه قبل ما تكمّل.
> - لو الاثنين > 0 → تمام، انتقل للفحص التحت.

**1) اكشف أي residue بروتينية ناقصة ذرة `CA`:**
```bash
cd /path/to/your_project/gromacs_run
awk '/^ATOM/ {
  key=substr($0,18,3)"|"substr($0,22,1)"|"substr($0,23,4);
  atom=substr($0,13,4); gsub(/ /,"",atom);
  a[key]=a[key]" "atom
}
END { n=0;
  for (k in a) if (a[k] !~ /(^| )CA( |$)/) { print "  ناقصة CA: "k"  ->  ذراتها:"a[k]; n++ }
  if (n==0) print "  ✓ ماكو residue مكسورة — complex.pdb سليم، انتقل للخطوة 4 مباشرة."
}' complex.pdb
```

**2) لو طلعت residues مكسورة — نظّفها (مع نسخة احتياطية):**
```bash
cp complex.pdb complex_backup.pdb

# اجمع مفاتيح الـ residues المكسورة (ناقصة CA)
awk '/^ATOM/ {
  key=substr($0,18,3)"|"substr($0,22,1)"|"substr($0,23,4);
  atom=substr($0,13,4); gsub(/ /,"",atom);
  a[key]=a[key]" "atom
}
END { for (k in a) if (a[k] !~ /(^| )CA( |$)/) print k }' complex.pdb > .broken_res.txt

echo "الـ residues الي راح تنحذف:"; cat .broken_res.txt

# احذف سطور تلك الـ residues (البروتين فقط — الليكند HETATM يبقى كما هو)
awk 'NR==FNR{bad[$0]=1; next}
{
  if ($0 ~ /^ATOM/) {
    key=substr($0,18,3)"|"substr($0,22,1)"|"substr($0,23,4);
    if (key in bad) next
  }
  print
}' .broken_res.txt complex.pdb > complex_clean.pdb

# اجعل السكربت يستعمل الملف النظيف بدون أي تعديل على run_server.sh
mv complex_clean.pdb complex.pdb
rm -f .broken_res.txt
echo "✓ تم التنظيف. النسخة الأصلية محفوظة بـ complex_backup.pdb"
```

**3) تأكد إن التنظيف صار عدل (لازم يطلع ✓ وإن الليكند لسه موجود):**
```bash
awk '/^ATOM/ {
  key=substr($0,18,3)"|"substr($0,22,1)"|"substr($0,23,4);
  atom=substr($0,13,4); gsub(/ /,"",atom); a[key]=a[key]" "atom
}
END { n=0; for(k in a) if(a[k]!~/(^| )CA( |$)/){print "  لسه مكسورة: "k; n++}
  if(n==0) print "  ✓ complex.pdb صار سليم — جاهز للتشغيل." }' complex.pdb
echo "عدد ذرات الليكند (HETATM): $(grep -c '^HETATM' complex.pdb)"
```

> **ملاحظة علمية:** الحذف مناسب للـ termini المكسورة أو الأطراف الناقصة (وهي بعيدة عادةً عن الموقع الفعّال). **لكن** لو الـ residue المكسورة داخل/قريبة من الـ binding site، **لا تحذفها** — بدلها أكمل ذراتها الناقصة بأداة مثل `pdbfixer` أو `modeller` قبل التشغيل.

---

## الخطوة 4 — شغّل السكربت داخل جلسة tmux (يحمي من قطع الاتصال)

مرحلة SwissParam + التحضير تشتغل على الـ **login node** وتنتظر (تستطلع كل دقيقتين)، فاستعمل `tmux` حتى ما تتوقف لو انقطع اتصالك:

```bash
tmux new -s md
bash run_server.sh
```

السكربت بالترتيب:
1. يستخرج البروتين/الليكند + **SwissParam** (يحتاج إنترنت — متوفر على الـ front node).
2. يبني النظام كامل (`pdb2gmx` CHARMM27 → دمج الليكند → صندوق → مذيب → أيونات 0.1 M → قيود → `index.ndx` بالاسم).
3. يولّد `md_gpu.slurm` ويرسله بـ `sbatch`، ويطبع **JobID**.

> فصل الجلسة بدون إيقاف: `Ctrl+b` ثم `d`. للرجوع لها: `tmux attach -t md`.

---

## الخطوة 5 — تابع وظيفة الـ GPU

```bash
squeue -u YOUR_USERNAME                 # حالة الوظيفة (PD=منتظرة، R=شغّالة)
tail -f md_<JobID>.out            # اللوق الحيّ (بدّل <JobID> بالرقم المطبوع)
```
الوظيفة تشغّل على الـ GPU بالترتيب: **EM → NVT → NPT → MD 100 ns**. تنتهي لمّا تشوف `### DONE ###`.

**تأكد أنها فعلاً على GPU:** داخل `md_<JobID>.out` لازم تشوف أسطر مثل `Using ... GPU(s)` و`offload`.

---

## الخطوة 6 — نزّل النتائج للاب توب

**على اللاب توب** (نافذة جديدة)، داخل مجلد المشروع:

```bash
# عدّل REMOTE_DIR داخل pull_results.sh ليطابق:  /path/to/your_project/gromacs_run
nano pull_results.sh
bash pull_results.sh              # المجلد كامل
# أو الأساسيات فقط:
bash pull_results.sh essentials
```

المخرجات النهائية: `MD.xtc` · `MD.gro` · `MD.tpr` · `MD.edr` · `MD.log` · `MD.cpt`.

---

## ملخّص الأوامر (نسخ سريع)

```bash
# --- على اللاب توب ---
rsync -avP ./ YOUR_USERNAME@HPC_LOGIN_NODE:/path/to/your_project/gromacs_run/
ssh YOUR_USERNAME@HPC_LOGIN_NODE

# --- على السيرفر ---
cd /path/to/your_project/gromacs_run
module avail 2>&1 | grep -i gromacs ; sinfo -s ; command -v obabel
nano run_server.sh          # طابق GMX_MODULE / SLURM_PARTITION / SLURM_GRES

# فحص إلزامي 0: تأكد complex.pdb فيه بروتين + ليكند (الاثنين لازم > 0) — راجع الخطوة 3.5
echo "ATOM=$(grep -c '^ATOM' complex.pdb)  HETATM=$(grep -c '^HETATM' complex.pdb)"

# فحص إلزامي 1: أي residue ناقصة CA (يمنع فشل pdb2gmx) — راجع الخطوة 3.5
awk '/^ATOM/{k=substr($0,18,3)"|"substr($0,22,1)"|"substr($0,23,4);t=substr($0,13,4);gsub(/ /,"",t);a[k]=a[k]" "t}
END{n=0;for(k in a)if(a[k]!~/(^| )CA( |$)/){print "  ناقصة CA: "k;n++};if(n==0)print "  ✓ complex.pdb سليم"}' complex.pdb
# لو طلعت أي residue مكسورة، نفّذ بلوك التنظيف بالخطوة 3.5 قبل ما تكمّل

tmux new -s md
bash run_server.sh
squeue -u YOUR_USERNAME ; tail -f md_<JobID>.out

# --- بعد الانتهاء، على اللاب توب ---
bash pull_results.sh
```

---

### ملاحظات دقّة
- **الملفات المولّدة مو موجودة قبل التشغيل:** `REC.pdb` · `LIG1.pdb` · `topol.top` · `conf.gro` · `md.log` ... كلها **يولّدها السكربت** — غيابها قبل التشغيل **طبيعي مو خطأ**. وتأكد دائماً إنك بمسار المشروع (`/path/to/your_project/gromacs_run`) مو بالـ `~/`.
- **`complex.pdb` لازم يحتوي بروتين (ATOM) + ليكند (HETATM):** لو `HETATM = 0` السكربت يوكف بـ `ERROR: ماكو HETATM (ليكند)`. افحص بـ `grep -c '^HETATM' complex.pdb` قبل التشغيل (الخطوة 3.5).
- **إلزامي قبل التشغيل (الخطوة 3.5):** افحص `complex.pdb` من أي residue ناقصة `CA`. لو موجودة، `pdb2gmx` يطيح بخطأ `atom CA ... not found`. السبب بالملف نفسه مو بالسكربت — نظّفه أول بأوامر الخطوة 3.5.
- تستعمل **ملفاتك الخاصة** للـ mdp — `tc-grps = Protein_LIG Water_and_ions` فيها يطابق `index.ndx` اللي يبنيه السكربت بالاسم، فما راح يطلع خطأ المجموعات.
- لو أيوناتك مو `NA`/`CL`، عدّل قائمة `resname` بسطر `gmx select` داخل `run_server.sh`.
- لو رقم الليكند/الاسم اختلف، ما يأثّر — التحضير يقرأ اسم `moleculetype` تلقائياً و`index.ndx` يُبنى بالاسم.
- SwissParam يشتغل على الـ **login node** لأنه يحتاج إنترنت؛ كل خطوات `mdrun` على الـ **GPU node** عبر SLURM.

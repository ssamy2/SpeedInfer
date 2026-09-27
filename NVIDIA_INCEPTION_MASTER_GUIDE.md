# SpeedInfer — دليل تقديم صادق إلى NVIDIA Inception

مراجعة 2026-09-27. يحل هذا النص محل الإرشادات السابقة التي ادعت قبولًا مضمونًا أو أوصت بوعود تقنية غير مثبتة.

## الأهلية

تذكر [NVIDIA رسميًا](https://www.nvidia.com/en-us/startups/) أن الشركة يجب أن تكون مسجلة رسميًا، عمرها أقل من عشر سنوات، لديها مطور واحد على الأقل وموقع يعمل. لا يشترط وجود إيرادات أو استخدام GPU من NVIDIA مسبقًا. تشمل الاستثناءات مقدمي الخدمات السحابية وإعادة البيع والاستشارات وغيرها؛ التصنيف النهائي تقرره NVIDIA.

توضح [دراسة Baseten الرسمية](https://www.nvidia.com/en-us/case-studies/baseten/) سابقة لمنصة استدلال ضمن منظومة Inception. وجود سابقة قريبة لا يضمن قبول SpeedInfer ولا يلغي شروط البرنامج. صِف المنتج كما يعمل: منصة برمجية لخدمة النماذج وواجهات API، وتدريب عند توفر التنفيذ الحقيقي، لا تغيير المصطلحات لإخفاء النشاط.

## صياغة المنتج الآن

> SpeedInfer is developing a managed model platform with an OpenAI-compatible text inference gateway, authenticated model access, token usage accounting, and a preview workspace for datasets, weights, training and deployment workflows. Live inference depends on connected model workers. Workspace training and deployment are currently simulations while execution infrastructure is being integrated.

هذه الصياغة تخص النسخة التي تمت مراجعتها؛ تُحدث عندما يثبت عمل الميزات الجديدة.

## إثبات المنتج

- جهّز هوية الشركة وتاريخ تسجيلها وبيانات الفريق وعرضًا موجزًا يشرح القيمة البرمجية المميزة.
- اعرض `/guide` وأمثلة تعمل بمفتاح تقييم محدود، ورصيد معلوم، وخادم نموذج متصل؛ لا تنشر سر المفتاح داخل المستودع أو العرض.
- أثبت نجاح طلب فعلي وبث مع usage، ثم رفض مفتاح بلا رصيد. استخدم `/ready` لفصل جاهزية العامل عن حيوية الموقع.
- اعرض المحاكاة باسمها، ولا تصف رفع أوزان بأنه تدريب مكتمل.
- لا تدّع عضوية أو اعتمادًا، أو عتاد Hopper/Blackwell، أو TensorRT/FP8، أو SLA، أو ZDR، أو أرقام TTFT دون تحقق وتوثيق فعلي.
- لا يوجد أسلوب تصميم يضمن قبول البرنامج. وضوح المنتج والمعلومات الحقيقية أقوى من وعود لا يستطيع المقيم إعادة اختبارها.

انظر [تقرير جاهزية الإنتاج](docs/PRODUCTION_READINESS_AR.md) للقيود وخطة تشغيل GPU والتحقق.

# Corpus de cuentos libres para la app de cuentos con IA (guía de licencias)

> Para qué sirve: definir qué cuentos escritos por personas podemos usar como **ejemplos de estilo, RAG,
> fine-tuning o semillas de escenarios** en una app comercial (SpA, pensada para venderse) sin
> problemas de derecho de autor.
> Investigado el 09-11-2026. **No se descargó ningún dataset ni se contactó a nadie.**
> Marcas: **[V]** = verificado en la fuente primaria o en un sitio oficial durante esta investigación;
> **[NV]** = no verificado, inferencia o dato de terceros; hay que confirmarlo antes de usarlo.
> Esto no es asesoría legal. Antes de vender la empresa (due diligence), un abogado de PI debe revisar el ledger.

---

## 1. Resumen ejecutivo

1. **Lo más seguro y útil:** StoryWeaver (CC BY 4.0, todo el catálogo, miles de títulos en español),
   African Storybook (CC BY 4.0), Global Digital Library (principalmente CC BY / CC BY-SA), textos en
   dominio público de Project Gutenberg sin la marca, y Wikisource (los textos en dominio público, no
   las traducciones nuevas hechas por voluntarios).
2. **Para lo chileno:** Ramón Laval, *Cuentos populares en Chile* (1923): el autor murió en 1929, así
   que está en dominio público. La BND lo marca como "Patrimonio cultural común" [V]. Es la mejor
   semilla folclórica chilena que encontré.
3. **Gabriela Mistral: no usarla por ahora.** Las fuentes se contradicen (ver §4).
4. **Hay que evitar:** licencias NC (no comercial) y ND (sin derivadas), las traducciones modernas de
   Grimm, Andersen, Perrault o Esopo, autores vivos o muertos hace menos de 70 años (Saúl Schkolnik
   murió en 2017 y su obra está protegida hasta 2087) y los datasets sintéticos generados con GPT
   (TinyStories, por ejemplo) para entrenar el modelo propio.
5. **Chile no tiene excepción de minería de textos y datos (TDM).** El art. 71 T se propuso en
   abril de 2026 y el Ejecutivo no lo repuso en julio de 2026. Para obras protegidas se necesita
   licencia, así que el corpus tiene que salir de dominio público o de licencias libres que permitan
   uso comercial.

---

## 2. Tabla de fuentes

| # | Fuente | URL | Licencia | ¿Comercial? | Atribución | ShareAlike / riesgo | Idioma | Volumen aprox. |
|---|---|---|---|---|---|---|---|---|
| 1 | **Project Gutenberg** | gutenberg.org/policy/license.html | Texto sin restricción de copyright en EE.UU. + **marca y licencia PG** [V] | **Sí**, si quitas la licencia PG y toda mención a "Project Gutenberg" del texto [V] | No es obligatoria para el texto PD. Si usas la marca comercialmente, se paga regalía (20%) [V para la regalía según la licencia; % NV] | Ninguno para el texto PD. **Ojo:** "PD en EE.UU." no siempre significa PD en Chile o la UE; hay que revisar la muerte del autor | EN (mucho), ES (cientos) | ~75.000 libros en total [NV]; incluye Grimm, Andersen, Esopo, Perrault, Lang y Laval (#63424) [V] |
| 2 | **Wikisource ES/EN** | es.wikisource.org · en.wikisource.org/wiki/Wikisource:Copyright_policy | Obra **PD**, o bien **CC BY-SA 4.0** para aportes originales, **traducciones de voluntarios incluidas** [V] | Sí | Si el texto es PD, no hace falta. Si es CC BY-SA, hay que atribuir | **Las traducciones de voluntarios son CC BY-SA.** Según la postura conservadora de CC, el modelo o los outputs que se publiquen tendrían que llevar la misma licencia [V, CC] → **usar solo los textos PD** | ES, EN | Miles de textos. Samaniego e Iriarte completos [V Samaniego], Calleja [V autor PD], Quiroga *Cuentos de la selva* [V] |
| 3 | **Biblioteca Nacional Digital / Memoria Chilena / Chile para Niños** | bibliotecanacionaldigital.gob.cl · memoriachilena.gob.cl | Cada documento indica su estatus: "Patrimonio cultural común" u "Obra protegida" [V] | **PD: sí**, "para los fines que estime conveniente". Protegida: no [V] | Se exige citar autor, título y fuente [V] | Ninguno en PD. Las obras protegidas solo se pueden citar o usar en enseñanza o investigación [V] | ES (Chile) | Miles de documentos. Para cuentos: Laval 1923 (PD) [V] |
| 4 | **Biblioteca Virtual Miguel de Cervantes** | cervantesvirtual.com/marco-legal | Condiciones restrictivas: solo uso personal y privado, **sin finalidad comercial**. Cualquier otro uso requiere autorización escrita [V] | **No** con sus archivos. La obra PD se puede sacar de otra edición | — | Sus **ediciones críticas modernas** (notas, fijación del texto) están protegidas [V, p. ej. la edición de Samaniego de E. Palacios] | ES | Muy grande |
| 5 | **Internet Archive** | archive.org · help.archive.org/help/rights | Sin licencia propia. Cada ítem trae su marca (p. ej. "Public Domain Mark 1.0") y el usuario asume el riesgo [V] | Solo si el ítem es realmente PD (hay que verificar al autor, no fiarse de la etiqueta) | Según el ítem | Hay errores de etiquetado y préstamos de libros protegidos. **Verificar caso por caso** | ES, EN | Enorme. Ej.: Quiroga 1918 con PDM [V] |
| 6 | **StoryWeaver (Pratham Books)** | storyweaver.org.in/en/open-content | **CC BY 4.0 en todo el contenido** [V] | **Sí** ("even commercially") [V] | Acreditar a Pratham Books, al donante o financista, y al autor, ilustrador y traductor [V] | Sin SA. **Bajo riesgo** | 370+ idiomas, con etiqueta Spanish [V] | ~66.000 cuentos en total (2025) [NV, fuente secundaria]. En español: miles [NV] |
| 7 | **African Storybook (Saide)** | africanstorybook.org | CC BY 4.0 [V, pie del sitio] | Sí | Autor, ilustrador, traductor y ASb | Sin SA. Revisar el pie de cada libro por si hay excepciones [NV] | Idiomas africanos, EN, FR, PT. **Poco español** [NV] | ~4.269 libros y ~8.098 traducciones (2020) [V, perfil DPG] |
| 8 | **Global Digital Library** | digitallibrary.io/about/license | "Primary licenses": **CC BY y CC BY-SA** [V] | Sí | Cada ítem viene marcado con su licencia y autores [V] | **Filtrar y descartar los CC BY-SA** (o aislarlos) | EN más idiomas de África y Asia | 500+ archivos imprimibles en 23 idiomas (2020) [V GitHub]. Total NV |
| 9 | **Bloom Library (SIL)** | bloomlibrary.org · bloom.sil.org/terms | **Mixta, por libro**: CC BY, CC BY-NC, CC BY-NC-ND, etc. [V] | **Solo** los libros CC BY o CC0 | Según el libro | Hay muchos libros **NC/ND** (p. ej. "© SIL Cameroon, CC BY-NC") [V] → filtrar uno por uno | Cientos de idiomas, incluido el español | Miles [NV] |
| 10 | **Let's Read (The Asia Foundation)** | letsreadasia.org | **Mixta, por libro**: hay libros con CC-BY-NC-4.0 [V]. Otros sitios del programa hablan de "CC Attribution" en general [V] | Solo los marcados CC BY | The Asia Foundation más un enlace [V] | Filtrar las NC. Muchos títulos se reciclan de StoryWeaver (CC BY) | Idiomas asiáticos, EN. Español escaso [NV] | Miles [NV] |
| 11 | **TinyStories (HF)** | huggingface.co/datasets/roneneldan/TinyStories | **CDLA-Sharing-1.0** [V] | La licencia lo permite, pero ver la columna de riesgo | Sí | (a) **Sintético**: generado por GPT-3.5 y GPT-4 [V], no escrito por personas. (b) CDLA-**Sharing** es copyleft sobre los datos. (c) Los términos de OpenAI restringen usar outputs para entrenar modelos que compitan con OpenAI [V, que la restricción existe; NV, su alcance sobre terceros] | EN | ~2M cuentos [NV] |
| 12 | **Children-Stories-Collection (ajibawa-2023)** | huggingface.co/datasets/ajibawa-2023/Children-Stories-Collection | Apache-2.0 [V] | Sí, según la etiqueta | Sí | **Sintético** [V] y con modelo generador no declarado. Mismo problema que el #11. No sirve como "escrito por humanos" | EN | ~0,9M [V] |
| 13 | **Hemeroteca y Biblioteca Digital Hispánica (BNE)** | bnelab.bne.es | Uso comercial gratuito de las imágenes de obras PD (desde 2020) [V, fuente secundaria] | Sí, en PD | Citar a la BNE [NV] | Bajo riesgo. Útil para ediciones antiguas de Calleja | ES | Grande |

---

## 3. Autores y obras: estado de dominio público

Regla chilena: la protección dura la vida del autor más 70 años (Ley 17.336 art. 10; Memoria Chilena) [V].
En la UE también son 70 años. **España** da 80 años a los autores muertos antes del 07-12-1987 [NV,
régimen transitorio]. EE.UU.: lo publicado antes de 1931 es PD (al 01-01-2026) [V, IA blog].

| Obra / autor | Muerte | Original | Traducciones utilizables | Notas |
|---|---|---|---|---|
| Hnos. Grimm | 1859/1863 | PD [V-hecho] | EN: Lucy Crane (1882), Margaret Hunt (1884) PD [NV]. ES: traducciones anónimas del s. XIX o de Calleja en Wikisource o Gutenberg | **No** usar traducciones editoriales actuales (Anaya, Alianza, etc.) |
| H. C. Andersen | 1875 | PD | EN: H. P. Paull (1872), Andrew Lang PD [NV]. **Jean Hersholt (1949) no** [NV]. ES: ediciones antiguas en Wikisource | |
| Charles Perrault | 1703 | PD | EN: Andrew Lang, *Blue Fairy Book* (1889) PD [NV] | |
| Esopo / Esopo | antigüedad | PD | EN: Townsend (1867), Jacobs (1894) PD [NV]. ES: Samaniego e Iriarte, que son reescrituras PD | |
| **Félix M. Samaniego** | 1801 | PD en todo el mundo [V, Wikisource] | — | Wikisource, edición de 1882 [V]. **No** usar la edición crítica de Cervantes Virtual |
| **Tomás de Iriarte** | 1791 | PD [V-hecho] | — | Wikisource |
| **Saturnino Calleja** | 1915 | PD [V, Wikisource] | "Las traducciones de sus obras pueden no estar en PD" [V] | Cuentos de Calleja |
| Fernán Caballero | 1877 | PD [NV] | — | Cuentos populares andaluces |
| **Horacio Quiroga**, *Cuentos de la selva* (1918) | 1937 | PD en Chile (1937+70 = 2007) y en EE.UU. (1918) [V, Wikisource "80 p.m.a." e IA PDM] | — | Excelente para niños, con animales |
| Rafael Pombo | 1912 | PD [NV] | — | Fábulas en verso (Colombia) |
| **Ramón A. Laval**, *Cuentos populares en Chile* (1923) | 1929 | **PD** ("Patrimonio cultural común", BND) [V] | — | También en Gutenberg #63424 y en LibriVox [V] |
| Julio Vicuña Cifuentes (mitos y supersticiones) | 1936 | PD en Chile [NV, fecha a confirmar] | — | Folclore chileno |
| Yolando Pino Saavedra (cuentos folclóricos chilenos) | 1992 | **Protegido** hasta 2062 [NV, fecha] | — | Aunque el cuento oral es libre, **su redacción y su compilación** están protegidas |
| **Saúl Schkolnik** | 25-10-2017 [V] | **Protegido hasta el 31-12-2087** | — | **No usar**, tampoco sus versiones de leyendas chilenas |
| **Gabriela Mistral** | 10-01-1957 | **En disputa** (ver §4) | — | No usar hasta que esté claro |
| Leyendas o mitos orales (Caleuche, Trauco, etc.) | — | La **trama folclórica** no tiene autor | — | Se pueden reescribir desde cero. **No** copiar ninguna versión moderna publicada |

---

## 4. Gabriela Mistral: estado en disputa

- **En contra de que sea PD hoy:** La Tercera (20-12-2025) y Diario El Día (24-01-2026) dicen que
  "desde 2028" pasa a dominio público (1957 + 70) y que hay diputados de la Comisión de Cultura que
  quieren **extender** la protección [V, que esas notas lo dicen].
- **A favor de que ya sea PD:** Wikimedia Chile y El Mostrador (09/10-08-2026) afirman que la obra
  "ya es parte del dominio público desde 1988" y que el **Conservador del Registro de Propiedad
  Intelectual lo ratificó en enero de 2026** [V, que el artículo lo dice. **NV** la resolución misma].
- **En la UE** queda protegida hasta el 01-01-2028 en cualquier caso [NV].
- **Regla:** no incluirla en el corpus antes del **01-01-2028** y vigilar el proyecto de extensión. Si
  hace falta antes, pedir al Conservador copia del pronunciamiento y guardarla en el ledger.

---

## 5. Creative Commons: qué se puede usar en una app comercial

| Licencia | Corpus comercial | Implicancia para el modelo y los outputs |
|---|---|---|
| **CC0 / PDM / PD** | ✅ Ideal | Ninguna |
| **CC BY 4.0** | ✅ Sí | Atribución (página de créditos). Sin copyleft |
| **CC BY-SA** | ⚠️ Evitar o aislar | CC, en su postura conservadora (2025), dice que **el modelo y los outputs que se publiquen deberían llevar la misma licencia** [V, guía CC "Using CC-licensed Works for AI Training"]. Choca con un producto cerrado y con su venta |
| **CC BY-NC (-SA)** | ❌ No | Una SpA que vende es uso comercial |
| **CC BY-ND / NC-ND** | ❌ No | Prohíbe adaptaciones. Un fine-tune o una reescritura puede contarse como adaptación |
| **CDLA-Sharing** | ⚠️ Evitar | Copyleft sobre los datos |
| **Apache-2.0 / MIT (datasets)** | ✅ si los datos son humanos y la fuente es legítima | Verificar el **origen** real del contenido, no solo la etiqueta |

Notas de CC [V, FAQ de CC]: las licencias solo operan donde el uso necesita permiso según el
derecho de autor. Los datos personales, la imagen y los contratos del sitio de origen corren por
cuerda separada.

---

## 6. Marco legal TDM (minería de textos y datos)

- **Chile [V]:** sigue la regla general de autorización previa (art. 19 de la Ley 17.336). Las
  excepciones que se suelen invocar (71 B cita, 71 O reproducción provisional, 71 Q uso incidental)
  **no calzan** con el entrenamiento masivo (Authedra, actualizado al 03-08-2026). El **art. 71 T**
  (excepción TDM amplia, art. 8 del proyecto de Reconstrucción Nacional, abril de 2026) generó
  rechazo. El 10-07-2026 Hacienda envió indicaciones **sin reponerlo** (The Clinic). **Estado
  actual: no hay excepción TDM vigente** [V a julio-agosto 2026. Revisar la tramitación antes del lanzamiento: NV desde esa fecha].
- **UE [V]:** la Directiva 2019/790, art. 4, permite TDM comercial sobre obras accesibles
  lícitamente **salvo opt-out** del titular. El AI Act exige respetar ese opt-out y publicar un resumen
  del contenido de entrenamiento si se ofrece un modelo de propósito general [NV, en detalle].
- **EE.UU. [V]:** el Copyright Office, *Part 3: Generative AI Training* (pre-publicación del
  09-05-2025), aplica fair use caso por caso y sin regla general. Los fallos son mixtos [NV, detalle].
- **Conclusión práctica:** con un corpus **PD + CC BY**, la pregunta TDM deja de importar en las tres
  jurisdicciones.

---

## 7. Reglas operativas

1. **Prioridad de licencias:** PD/CC0 > CC BY > (nada más). NC, ND, SA y CDLA-Sharing quedan fuera
   del corpus de entrenamiento. Si alguna vez se usan, que sea solo como referencia interna no
   distribuida, y con el visto bueno de un abogado.
2. **PD depende de la jurisdicción:** hay que revisar el año de muerte del **autor y del traductor**
   (y del ilustrador, si se usan imágenes). Un texto se acepta si es PD en **Chile, EE.UU. y la UE**
   a la vez.
3. **Traducciones:** solo se aceptan traducciones antiguas con traductor muerto hace más de 70 años, anónimas del s. XIX o con
   licencia CC BY. **Lo más seguro es que nosotros traduzcamos el original PD** (la traducción queda como
   obra propia).
4. **Project Gutenberg:** al ingerir, se eliminan el header, el footer y toda mención a "Project
   Gutenberg". Nunca usar la marca en la app ni en el marketing. Un agradecimiento con enlace en los
   créditos sí está permitido [V].
5. **Ediciones críticas modernas** (Cervantes Virtual, editoriales actuales): no se usan. Hay que ir a
   la edición antigua digitalizada (BNE, BND, IA o Gutenberg).
6. **Datasets sintéticos** (TinyStories y similares): fuera del corpus "humano". No sirven para el
   argumento de venta ("escrito por personas") y arrastran los términos del modelo que los generó.
7. **Folclore:** se toma la trama y se reescribe con voz propia. Nunca se copia la redacción de una
   compilación moderna.
8. **Outputs:** filtro anti-regurgitación (n-gramas largos contra el corpus) para que la app no
   devuelva un texto ajeno casi literal. Mostrar como "inspirado en" en vez de reproducir.
9. **Ledger obligatorio** (abajo): sin una fila en el ledger, el texto no entra al pipeline. Es la
   pieza clave de la due diligence al vender la SpA.
10. **Página de créditos en la app** ("Créditos y licencias"), generada desde el ledger: título,
    autor, ilustrador, traductor, fuente, licencia con enlace y nota "adaptado". Para StoryWeaver
    también se acredita a Pratham Books y al financista [V].
11. **Snapshot:** guardar el HTML o PDF de la página de licencia al momento de la descarga (hash +
    fecha). Las licencias de cada libro pueden cambiar.

### 7.1 Esquema del ledger (`licencias_corpus.csv`)

```
id, titulo, autor, autor_muerte, traductor, traductor_muerte, ilustrador,
fuente_nombre, fuente_url, url_licencia, licencia_spdx,      # p.ej. CC-BY-4.0, CC0-1.0, PDM, PD-CL
pd_chile, pd_eeuu, pd_ue,                                    # si/no/na
idioma, palabras, uso_permitido,                             # entrenamiento|rag|semilla|solo_referencia
atribucion_texto,                                            # cadena exacta para créditos
modificado,                                                  # si/no + descripcion
sintetico,                                                   # si/no (y modelo generador)
fecha_descarga, sha256_archivo, snapshot_licencia_path,
revisado_por, fecha_revision, notas
```

Reglas del ledger: `licencia_spdx` vacía o con NC/ND significa **exclusión automática**.
`pd_* = no` en cualquier jurisdicción significa exclusión. `sintetico = si` va a un pool separado.

---

## 8. Plan mínimo sugerido (opciones)

1. **Arranque rápido (recomendado):** StoryWeaver en español y EN (CC BY) + Laval + Quiroga + Samaniego +
   Iriarte + Calleja + Grimm/Andersen/Perrault/Esopo en traducciones PD de Gutenberg o Wikisource.
   Entra todo al ledger y se usa como **RAG y ejemplos de estilo**, no como fine-tuning, al principio.
2. **Más volumen:** sumar African Storybook y GDL (solo CC BY), y Bloom y Let's Read filtrados por
   libro.
3. **Diferenciación chilena:** encargar cuentos originales a autores chilenos con **contrato de cesión**
   o licencia expresa para entrenamiento de IA. Es lo único que da voz chilena moderna sin riesgo.

---

## 9. Datasets en Kaggle y Hugging Face

> Revisado el 09-11-2026 leyendo las fichas públicas (API pública de Hugging Face y de Kaggle, más el
> README de cada dataset). **No se descargó nada ni se creó ninguna cuenta.**
> Regla de oro: **la etiqueta de licencia la pone quien sube el archivo, no el autor del texto.** En
> Kaggle hay datasets marcados "CC0" con obras protegidas (ver Walsh). Lo que manda es el **origen
> real del texto**, igual que en §5.

**Hallazgo principal:** en español casi **no existen** datasets de cuentos escritos por personas con
licencia clara. Lo útil en español sigue siendo armar el corpus nosotros desde Gutenberg, Wikisource,
StoryWeaver y la BND (§2). En inglés sí hay un par de datasets aprovechables, todos derivados de
Project Gutenberg.

### 9.1 Hugging Face

| Dataset | URL | Licencia (etiqueta) | ¿Fine-tuning comercial? | Origen del texto | Tamaño | Veredicto |
|---|---|---|---|---|---|---|
| sedthh/gutenberg_english | huggingface.co/datasets/sedthh/gutenberg_english | MIT [V] | Sí, el texto es PD en EE.UU. y la MIT cubre solo la recopilación | Gutenberg, metadatos quitados y textos marcados con copyright filtrados [V] | 48.284 libros [V] | **USAR** filtrando libros infantiles y revisando muerte del autor/traductor (§3) |
| sedthh/gutenberg_multilang | huggingface.co/datasets/sedthh/gutenberg_multilang | MIT [V] | Sí, mismas condiciones | Gutenberg en ES, DE, FR, NL, IT, PT, HU [V] | 7.907 libros en total (~3,1 GB) [V]; cuántos en español: NV | **USAR** solo el subconjunto `es`, filtrando cuentos. Es la mejor fuente en español de esta lista |
| manu/project_gutenberg | huggingface.co/datasets/manu/project_gutenberg | Sin etiqueta; remite a la licencia de PG [V] | Sí para el texto PD, si se quita el encabezado y el pie de PG (§7.4) | Gutenberg multilingüe, incluido `es` [V] | 10K–100K libros [V] | **USAR con cuidado**: hay que limpiar la marca PG |
| deepmind/pg19 | huggingface.co/datasets/deepmind/pg19 | Apache-2.0 [V] | Sí | Libros de Gutenberg publicados antes de 1919 [NV] | ~28.000 libros [NV] | **USAR** como reserva en inglés. Casi todo es para adultos, así que hay que filtrar |
| WorkInTheDark/FairytaleQA | huggingface.co/datasets/WorkInTheDark/FairytaleQA | Apache-2.0 [V] | Sí | 278 cuentos para niños con 10.580 preguntas anotadas por expertos [V]. Los cuentos vienen de Gutenberg [NV] | 278 cuentos [V] | **USAR** para evaluar la comprensión, no por volumen |
| cam-cst/cbt (Children's Book Test) | huggingface.co/datasets/cam-cst/cbt | **GFDL** [V] | ⚠️ GFDL es copyleft | Libros infantiles libres (Montgomery, Dickens, Lang…) [V] | 100K–1M ejemplos [V] | **NO USAR** el dataset. Mejor ir directo a los libros PD en Gutenberg |
| bookcorpus/bookcorpus | huggingface.co/datasets/bookcorpus/bookcorpus | "unknown" [V] | **No** | Scrapeado de Smashwords. Muchos autores licencian "solo para disfrute personal" [V, ficha] | 10M–100M frases [V] | **NO USAR** |
| euclaise/writingprompts | huggingface.co/datasets/euclaise/writingprompts | MIT [V] | No es confiable | Posts de usuarios de Reddit r/WritingPrompts [NV]. El copyright es de cada usuario | 100K–1M [V] | **NO USAR**: la etiqueta MIT no arregla el origen |
| roneneldan/TinyStories | huggingface.co/datasets/roneneldan/TinyStories | CDLA-Sharing-1.0 [V] | — | **Sintético, GPT-3.5 y GPT-4** [V] | 1M–10M [V] | **NO USAR** (rompe "100% humano") |
| robrenaud/multilingual_tinystories | huggingface.co/datasets/robrenaud/multilingual_tinystories | Apache-2.0 [V] | — | TinyStories en **español generado por código** [V]; modelo generador NV | ~5 millones de cuentos en español [V, README de GitHub] | **NO USAR** (sintético) |
| ajibawa-2023/Children-Stories-Collection | huggingface.co/datasets/ajibawa-2023/Children-Stories-Collection | Apache-2.0 [V] | — | **Sintético** [V] | ~0,9M [V] | **NO USAR** (sintético) |
| merve/folk-mythology-tales | huggingface.co/datasets/merve/folk-mythology-tales | Sin licencia [V] | No | Copia del sitio de D. L. Ashliman (sites.pitt.edu/~dash) [V]. Tiene traducciones y redacción propias con copyright [NV] | 100K–1M [V] | **NO USAR** el dataset. Sirve como índice para buscar los originales PD que cita |

### 9.2 Kaggle

| Dataset | URL (kaggle.com/datasets/…) | Licencia (etiqueta) | ¿Fine-tuning comercial? | Origen del texto | Tamaño | Veredicto |
|---|---|---|---|---|---|---|
| edenbd/children-stories-text-corpus | …/edenbd/children-stories-text-corpus | CC0 [V] | Sí | Libros infantiles **PD de Gutenberg, elegidos a mano y limpiados** (usado en FairyTailor) [V] | ~7,6 MB [V] | **USAR**. Es el mejor candidato en inglés. Igual hay que pasar cada título por el ledger |
| cuddlefish/fairy-tales | …/cuddlefish/fairy-tales | CC0 [V] | Sí, si el texto es PD | "English Fairy Tales from 19th to early 20th century" [V]. Sin fuente declarada | ~4,7 MB [V] | **USAR con verificación** título por título (las traducciones de comienzos del s. XX pueden no ser PD en Chile) |
| tschomacker/grimms-fairy-tales | …/tschomacker/grimms-fairy-tales | CC0 [V] | Sí | Texto de Gutenberg, con la licencia PG incluida [V] | ~0,2 MB [V] | **USAR** quitando la marca PG. Revisar qué traducción es (Hunt 1884, NV) |
| balazskocsis11/h-c-andersens-fairy-tales | …/balazskocsis11/h-c-andersens-fairy-tales | CC0 [V] | Depende de la traducción | Andersen en inglés; **traductor no declarado** [V] | ~0,1 MB [V] | **USAR solo si** se identifica una traducción PD. Si es Hersholt (1949): NO |
| pranayrishith16/aesops-fables-text-corpus | …/pranayrishith16/aesops-fables-text-corpus | CC0 [V] | Depende de la traducción | Esopo en inglés; traductor no declarado [V] | ~0,1 MB [V] | **USAR con verificación** (Townsend o Jacobs: sí) |
| chayanonc/1000-folk-stories-around-the-world | …/chayanonc/1000-folk-stories-around-the-world | CC0 [V] | Por fila | Trae columna `source`, pero no se sabe de dónde se copió [V] | ~4,6 MB [V] | **VERIFICAR fila por fila**. Si no se puede, NO USAR |
| planetskaro/fairy-tales-structured-dataset | …/planetskaro/fairy-tales-structured-dataset | CC0 [V] | No confiable | 4.317 cuentos "scrapeados" y procesados para fine-tuning [V]. **Fuente no declarada** | ~30 MB [V] | **NO USAR** hasta saber de dónde salió |
| martinagrunewald/maria-elena-walsh-obra-infantil | …/martinagrunewald/maria-elena-walsh-obra-infantil | **CC0 (falso)** [V la etiqueta] | **No** | Obra infantil de **María Elena Walsh** [V], que murió en 2011 [NV]. Protegida hasta ~2081 | ~0,4 MB [V] | **NO USAR**. Es el ejemplo perfecto de etiqueta CC0 sobre obra protegida |
| victoriaespinosa/cuentos | …/victoriaespinosa/cuentos | Unknown [V] | No | Sin descripción [V] | ~1 MB [V] | **NO USAR** |
| annbengardt/fairy-tales-from-around-the-world | …/annbengardt/fairy-tales-from-around-the-world | Unknown [V] | No | No declarado | ~7,3 MB [V] | **NO USAR** |
| thedevastator/folk-tales | …/thedevastator/folk-tales | "Other" [V] | No confiable | Cuentos por país, sin fuente clara [V] | ~11 MB [V] | **NO USAR** |
| andrzejpanczenko/folk-tales-dataset | …/andrzejpanczenko/folk-tales-dataset | "Contents: © Original Authors" [V] | No | Con copyright de terceros | ~9,8 MB [V] | **NO USAR** |
| mahendran1/bed-time-stories | …/mahendran1/bed-time-stories | "© Original Authors" [V] | No | Con copyright de terceros | <0,1 MB [V] | **NO USAR** |
| abhijeet000000007/ai-generated-stories | …/abhijeet000000007/ai-generated-stories | CC BY-NC-SA 4.0 [V] | No | Generado por IA [V, título] | ~3 MB [V] | **NO USAR** (NC y sintético) |
| ismaelfi/melolingua-cefr-graded-multilingual-stories | …/ismaelfi/melolingua-cefr-graded-multilingual-stories | CC BY-NC 4.0 [V] | No | No declarado | ~0,7 MB [V] | **NO USAR** (NC) |

**Cómo usar estos datasets:** no se cargan enteros. Se toman como **lista de candidatos**: cada
texto vuelve a pasar por el ledger (§7.1 y §11.3), con autor, traductor, año y fuente original
verificados. Si no se puede saber quién lo escribió, no entra.

---

## 10. Modelo base abierto

**Hardware gratis de Kaggle:** sesiones de hasta 12 h con GPU, **cuota de ~30 h por semana "o a
veces más"**, y opciones P100 (16 GB) o T4 x2 (2 × 16 GB, tarjetas separadas) [V, docs de Kaggle].
Un blog dice que Kaggle retiró la P100 el 15-09-2026 y dejó solo la T4 x2 [NV]. Para planificar,
conviene suponer **una T4 de 16 GB**: no tiene bf16 (se usa fp16) y obliga a hacer **QLoRA en
4 bits**. Con eso cabe cómodo un modelo de 3–8B; uno de 12B queda muy justo [NV, regla práctica].

| Modelo | Licencia | ¿Cabe en T4 con QLoRA? | Español | Ojo con |
|---|---|---|---|---|
| **Qwen3-4B / Qwen3-8B** (Alibaba) | **Apache-2.0** [V, etiqueta HF] | Sí (4B holgado, 8B justo pero cabe) [NV] | "100+ idiomas" [V, ficha]. Buen español [NV] | **No todos los Qwen son Apache:** Qwen2.5-**3B** usa la "Qwen RESEARCH LICENSE" [V], que no sirve para uso comercial. Hay que revisar la licencia **de cada tamaño** |
| **Gemma 4 E4B / 12B** (Google) | **Apache-2.0** [V, etiqueta HF y ficha] | E4B sí. 12B justo [NV] | Preentrenado en 140+ idiomas, 35+ con soporte directo [V] | **Gemma 3 y anteriores** usan los "Gemma Terms of Use" con política de usos prohibidos y descarga con aceptación previa (gated) [V, que está gated; NV el detalle]. Usar Gemma **4**, que ya es Apache |
| **Salamandra-7B-instruct** (BSC, España) | **Apache-2.0** [V] | Sí [NV] | Entrenado desde cero en 35 idiomas europeos, con **lenguas de España sobremuestreadas 2×** [V]. Europeo, pero sin sesgo a EE.UU. | Menos probado en tareas creativas [NV]. Español de España, no chileno: el fine-tune lo corrige |
| **Mistral-Nemo-Instruct-2407** (12B) | **Apache-2.0** [V] | Muy justo en una T4; mejor en 2 × T4 [NV] | Español declarado; MMLU en español 64,6% [V, ficha] | Más pesado que los demás. Mistral-7B-v0.3 (Apache [V]) es la versión liviana, pero tiene peor español [NV] |
| *(alternativa)* Llama-3.1-8B-Instruct (Meta) | Licencia propia "llama3.1" [V], gated [V] | Sí | Español declarado [V] | Obliga a mostrar **"Built with Llama"** y a **poner "Llama" al comienzo del nombre** de cualquier modelo derivado que se distribuya [V]. Pide licencia aparte sobre 700M usuarios/mes [V]. Tiene política de uso aceptable [NV, detalle]. **No es "open source" en sentido OSI** y choca con publicar el modelo como propio |

**Recomendación (opciones):**
1. **Qwen3-4B o Gemma 4 E4B**: los dos son Apache-2.0, caben holgados en una T4 y permiten iterar
   rápido con 30 h por semana. Además son "Small", que calza con Mozilla (§12.4).
2. **Salamandra-7B** si se prioriza un origen europeo con datos documentados (eso ayuda a la
   narrativa de transparencia).
3. Llama solo si hay una razón fuerte: sus condiciones de marca y nombre complican la licencia
   abierta del proyecto.

En todos los casos: guardar en el ledger el **hash de la versión exacta** del modelo base y copia del
archivo de licencia del día de descarga (igual que §7.11).

---

## 11. Afirmación "100% texto humano": cómo decirla con honestidad

### 11.1 Qué cubre y qué no
- **Cubre:** el **corpus con que NOSOTROS hacemos el fine-tuning** (y el de RAG o ejemplos de
  estilo): todo texto fue escrito por personas y tiene origen y licencia documentados.
- **No cubre:** el **preentrenamiento del modelo base** (Qwen, Gemma, etc.). No sabemos con qué
  datos exactos se entrenó y probablemente incluye texto generado por IA [NV]. Tampoco cubre los
  cuentos que genera la app: **esos los escribe una IA** y hay que decirlo.
- Nunca decir "nuestra IA es 100% humana" ni "modelo entrenado solo con texto humano". Eso sería
  falso.

### 11.2 Cómo se garantiza (en orden de fuerza)
1. **Fecha:** solo entran textos **publicados antes de 2022**. ChatGPT salió a fines de 2022 [NV,
   fecha], así que antes de eso casi no había texto generado por IA. Todo el dominio público
   (§3) cumple de sobra.
2. **Autoría verificada:** si el texto es de 2022 o después (StoryWeaver nuevo, encargos a autores
   chilenos), entra solo con **autor identificado** y, en los encargos, con una **declaración firmada
   de autoría humana** dentro del contrato de cesión.
3. **Ledger por texto** (§11.3): sin fila no entra.
4. **Detectores de IA: solo un filtro extra, nunca la garantía.** Tienen falsos positivos (marcan
   como IA textos humanos, sobre todo textos simples o de no nativos) y falsos negativos. OpenAI retiró
   su propio clasificador en 2023 por baja precisión [NV]. Se usan para **levantar alertas** que
   revisa una persona, no para aprobar ni rechazar.

### 11.3 Columnas extra para el ledger (`licencias_corpus.csv`)
Se agregan al esquema de §7.1 (no lo reemplazan):

```
fecha_publicacion_original,      # año o fecha de la primera publicación (clave: < 2022)
tipo_procedencia,                # dominio_publico | cc_by | encargo_con_contrato | propio_equipo
evidencia_autoria,               # p.ej. "edición 1923 BND id 624652", "contrato N°", "ficha StoryWeaver"
contrato_cesion_id,              # solo encargos; incluye cláusula de autoría humana
declaracion_humano_firmada,      # si/no/na
dataset_intermedio,              # si se encontró vía Kaggle/HF: id del dataset (solo como pista)
detector_ia_nombre_version,      # opcional; filtro extra
detector_ia_resultado,           # opcional; nunca decide solo
revision_humana_resultado,       # aprobado/rechazado + motivo
incluido_en_corpus_version       # p.ej. corpus-v0.3 (para reconstruir qué se usó en cada entrenamiento)
```

Regla nueva: `fecha_publicacion_original >= 2022` **y** `evidencia_autoria` vacía ⇒ exclusión
automática. `sintetico = si` ⇒ nunca entra al corpus "humano".

### 11.4 Texto sugerido para postulaciones (copiar tal cual)
**Español:**
> "El corpus que usamos para ajustar (fine-tuning) nuestro modelo está formado 100% por textos escritos
> por personas: cuentos en dominio público publicados antes de 2022 y obras con licencia abierta (CC BY)
> o encargadas a autores con contrato y declaración de autoría. Cada texto tiene registrada su fuente,
> autor y licencia en un registro público. No usamos textos generados por IA en ese corpus. El modelo
> base es [Qwen3-4B / Gemma 4], publicado por terceros con licencia Apache-2.0; su preentrenamiento no
> está bajo nuestro control y esta afirmación no lo cubre. Los cuentos que entrega la app son
> generados por IA y se presentan como tales."

**English:**
> "Our fine-tuning corpus is 100% human-written: public-domain stories published before 2022, openly
> licensed (CC BY) works, and commissioned stories with signed authorship declarations. Every text is
> recorded with its source, author and license in a public provenance ledger. No AI-generated text is
> included in that corpus. The base model ([Qwen3-4B / Gemma 4], Apache-2.0) was pre-trained by a third
> party; its pre-training data is outside our control and outside this claim. Stories produced by the
> app are AI-generated and labelled as such."

---

## 12. Biblia del multiverso

La idea: que cada temporada y episodio respete lo que ya pasó, y que cualquiera pueda reusar el
universo. Se guarda como **archivos de texto simples (YAML o JSON) en un repositorio**, uno por
entidad, para que se puedan revisar, versionar y publicar.

### 12.1 Estructura de carpetas
```
biblia/
  mundos/        el-bosque-de-lluvia.yaml
  personajes/    tomasa-la-zorra.yaml
  temporadas/    t01-el-invierno-largo.yaml
  episodios/     t01e03-la-cueva-del-eco.yaml
  canon/         reglas.yaml          # lo que NUNCA se puede contradecir
  variantes/     tomasa-espejo.yaml   # mismo personaje en otro mundo, a veces invertido
  LICENSE        CC BY 4.0
  CREDITOS.md    generado desde el ledger (§7.10)
```

### 12.2 Campos mínimos
```yaml
# personajes/tomasa-la-zorra.yaml
id: tomasa
nombre: Tomasa
especie: zorra culpeo
mundo_origen: bosque-de-lluvia
rasgos: [curiosa, miedosa de noche, inventa canciones]
nunca: [miente a su hermano]          # regla de personaje (canon)
aparece_en: [t01e01, t01e03, t02e05]
estado_actual: "vive en la cueva del eco desde t01e03"
inspirado_en: "trama folclórica del zorro culpeo; Laval 1923 (PD)"   # fuente del ledger, si aplica
licencia: CC-BY-4.0

# variantes/tomasa-espejo.yaml
id: tomasa-espejo
variante_de: tomasa
mundo: ciudad-de-niebla
inversion: [valiente de noche, miedosa de día]   # qué rasgos se dan vuelta
relacion_con_original: "se conocen en t03e07"

# episodios/t01e03-la-cueva-del-eco.yaml
id: t01e03
temporada: t01
mundo: bosque-de-lluvia
personajes: [tomasa, don-pancho]
resumen: "Tomasa descubre que la cueva repite los secretos."
cambios_de_estado: ["tomasa se muda a la cueva"]  # se copian a cada personaje
reglas_tocadas: [eco-solo-de-dia]
edad_objetivo: 4-7

# canon/reglas.yaml
- id: eco-solo-de-dia
  regla: "La cueva solo devuelve el eco de día."
  desde: t01e03
```

**Chequeo de coherencia (simple):** antes de publicar un episodio nuevo, un script revisa que
(a) los personajes existan, (b) ningún `cambios_de_estado` contradiga `estado_actual` ni una regla
de `canon/`, y (c) las variantes declaren qué invierten. El mismo YAML se le pasa al modelo como
contexto (RAG) al generar, para que no invente contra el canon.

### 12.3 Licencia para publicar nuestro multiverso
- **Contenido** (personajes, mundos, episodios, biblia): **CC BY 4.0** [NV, recomendación]. Permite que
  otros lo usen, incluso en forma comercial, siempre que nos citen. Es la misma licencia de StoryWeaver
  (§2), así que se mezcla sin conflicto con ese material.
- **Código** (scripts de coherencia, app de ejemplo): **Apache-2.0** (incluye licencia de patentes) o MIT.
- **Esquema de datos** (la plantilla YAML vacía): **CC0**, para que cualquiera lo copie sin trámite.
- **Pesos del modelo ajustado:** Apache-2.0 si el modelo base es Apache (Qwen3, Gemma 4, Salamandra).
  Con Llama habría que respetar su licencia y su regla de nombre (§10).
- **Ojo:** CC BY entrega los personajes. Un tercero puede hacer su propia app con Tomasa. Lo que
  protege a la SpA es la **marca** ("Roanso" y el nombre de la app, registrables en INAPI), porque las
  licencias CC no licencian marcas [NV, texto de la licencia]. Lo que no se quiera regalar (por ejemplo,
  la voz o la ilustración premium) se deja fuera de la biblioteca abierta desde el principio.
- **Cuidado con lo derivado:** si un episodio adapta un cuento CC BY, se mantiene la atribución
  original en `inspirado_en` y en CREDITOS.md. Si algo viniera de CC BY-SA, ese episodio quedaría
  obligado a CC BY-SA (por eso §7.1 lo excluye).

### 12.4 Cómo calza con Mozilla Human-Scale AI (cierre: 20-11-2026)
- **Plazo:** **20-11-2026** [NV: dato del encargo; la página oficial de Mozilla se bloqueó y el
  formulario no muestra la hora]. Si el cierre es a medianoche de EE.UU., en Chile ya sería el 21-11
  de madrugada. Conviene enviar el **19-11**.
- **Postulantes con fines de lucro:** el formulario permite "For-profit or other organization" [V]. Pero
  hay una pregunta de **propósito caritativo / beneficio público** [V], así que el proyecto debe mostrar
  ese beneficio (biblioteca abierta de cuentos para niños hispanohablantes).
- **Principios:** el formulario pide marcar **Small, Intentional, Tailored** [V]. Según una fuente
  secundaria, los proyectos deben cumplir los tres: modelos chicos que corran en equipos modestos,
  control real de usuarios y comunidad, y diseño para una necesidad concreta [NV, ImpactOpen]. Un modelo
  de 4B ajustado en GPU gratis calza con "Small".
- **Trabajo abierto:** el formulario pregunta "How do you plan to work in the open?" con opciones como
  código abierto, **compartir datasets abiertamente**, materiales educativos y publicaciones [V], y pide
  explicar cómo se compartirán "code, learnings, and data" [V]. Además buscan proyectos **ya
  desplegados con usuarios**, aunque sea un piloto chico [V].
- **Qué publicar para cumplir:**
  1. La **biblia del multiverso** y los episodios, con CC BY 4.0.
  2. El **ledger de procedencia** (§7.1 + §11.3) como dataset abierto. Es la prueba del "100% humano".
  3. El **corpus** solo en la parte que se pueda redistribuir: PD y CC BY con atribución. No incluye
     encargos con cesión que no permita redistribuir.
  4. Código Apache-2.0 y, si se puede, los pesos del LoRA.
- **Monto:** el formulario habla de una subvención de **US$300.000** [V]. Una fuente secundaria dice
  "hasta US$500.000 en tres años" [NV]. Confirmar en la convocatoria oficial (la página de Mozilla se
  bloqueó con Cloudflare durante esta investigación).

---

## Fuentes consultadas
- gutenberg.org/policy/license.html · /permission.html [V]
- en.wikisource.org/wiki/Wikisource:Copyright_policy; es.wikisource.org (Samaniego, Calleja, Quiroga) [V]
- bibliotecanacionaldigital.gob.cl (FAQ Propiedad intelectual); BND Visor 624652 (Laval) [V]
- memoriachilena.gob.cl/602/w3-article-622839.html (plazo 70 años) [V]
- cervantesvirtual.com/marco-legal [V]
- storyweaver.org.in/en/open-content [V]; africanstorybook.org [V]; digitalpublicgoods.net (ASb) [V]
- digitallibrary.io/about/license [V]; github.com/GlobalDigitalLibraryio/print [V]
- bloom.sil.org/terms; docs.bloomlibrary.org; community.software.sil.org/t/862 [V]
- letsreadasia.org (fichas por libro) [V]
- huggingface.co/datasets/roneneldan/TinyStories (README) [V]; ajibawa-2023/Children-Stories-Collection [V]
- openai.com/policies + help.openai.com/12092907 (restricción de entrenar modelos competidores) [V, que existe]
- creativecommons.org/faq; creativecommons.org/using-cc-licensed-works-for-ai-training-2 [V]
- authedra.ai/cl/recursos/derecho-autor-inteligencia-artificial-ley-17336 (03-08-2026) [V]
- theclinic.cl 10-07-2026 (71 T no repuesto) [V, titular y bajada]; biobiochile, lexlatin (71 T) [V, titulares]
- wikimedia.cl y elmostrador.cl (Mistral PD desde 1988) [V, que lo afirman]; latercera.com y diarioeldia.cl (Mistral 2028) [V, que lo afirman]
- es.wikipedia.org/wiki/Saúl_Schkolnik (1929-2017) [V]
- eur-lex 32019L0790 (art. 4) [V]; copyright.gov/ai Part 3 [V]
- huggingface.co/api/datasets/<id> y README de cada dataset de §9.1 (09-11-2026) [V]
- kaggle.com/api/v1/datasets/list y /metadata/<ref> para los datasets de §9.2 (09-11-2026) [V]
- huggingface.co/api/models (Qwen3, Gemma 4, Salamandra, Mistral, Llama 3.1: etiqueta de licencia y gated) [V]; LICENSE de Qwen2.5-3B-Instruct ("Qwen RESEARCH LICENSE") [V]
- raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_1/LICENSE ("Built with Llama", nombre "Llama…", 700M MAU) [V]
- ai.google.dev/gemma/terms y ai.google.dev/gemma/docs/gemma_4_license (Apache 2.0) [V]
- kaggle.com/docs/efficient-gpu-usage (cuota GPU ~30 h/semana) [V]
- mozillafoundation.tfaforms.net/f/humanscaleai (formulario: US$300.000, For-profit permitido, "work in the open", proyectos ya desplegados, propósito caritativo) [V]; página de convocatoria de mozillafoundation.org bloqueada por Cloudflare [no leída]; impactopen.org (resumen secundario) [NV]

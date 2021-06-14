py_oma:

Toolbox for operational modal analysis in python.

Ich habe Jupyter- Notebooks für SSi und FDD in /snippets gelegt. Diese müssten wir jetzt als Paket strukturieren und ich vllt nochmal überarbeiten. Bei den Paketen, die ich bisher geschrieben habe, habe ich es in der Regel so gehandhabt, dass ich ein zentrales Objekt zur Steuerung der verschiedenen Methoden angelegt habe. Weiß auch nicht wie smart das ist, aber funktioniert.

In unserem Fall würde ich Vorschlagen, dass es eine Klasse gibt OMA(). Die hat dann zentrale Funktionen, wie load_data(), detrend(), etc...

Der Workflow wäre dann quasi:

```python

# create pyoma object
oma = OMA()

# load and pre-process data
oma.load_data('path_to_file', fs)  # we should implement several interfaces to different file types and SQL

oma.detrend('linear')

# Modal analysis
f, ms = fdd(params)

f, ms, zeta = ssi_cov(params)

```

Was hälst du davon?


# ToDo:

+ [ ]  Struktur festlegen. Vielleicht habe ich schon bisschen oft OMA verwendet, sodass paKete und Funktionen nicht mehr so klar unterscheidbar sind.

+ [X] Tests schreiben

+ [X] einfache funktion für load_data schreiben, z.b. .csv (obsolet)

+ [X]  fdd übertragen

+ [ ]  ssi_cov nochmal prüfen und übertragen

+ [ ] testen

+ [ ]  100 Dinge, die noch nicht absehbar sind

+ [X]  setup schreiben

+ [X]  State-Space-Model für Test (Andi)



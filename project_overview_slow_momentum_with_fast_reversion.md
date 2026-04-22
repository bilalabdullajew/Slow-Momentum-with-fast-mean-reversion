# Project Overview — Slow Momentum with Fast Reversion

## Projektziel

Ziel dieses Projekts ist die saubere, reproduzierbare Nachbildung des **LSTM-CPD-Modells** aus dem Paper *Slow Momentum with Fast Reversion: A Trading Strategy Using Deep Learning and Changepoint Detection*.

Der Fokus liegt **ausschließlich** auf diesem Modell und seiner vollständigen Research- und Implementierungsstrecke:

- präzise Extraktion der Modellarchitektur und Methodik aus dem Paper,
- Überführung in eine belastbare technische Spezifikation,
- Ableitung eines konkreten Implementierungsplans,
- Zerlegung in atomare Tasks ohne Interpretationsspielraum,
- anschließende Umsetzung in Python,
- abschließende Bündelung der gesamten Forschung in einem reproduzierbaren **Jupyter Notebook**.

Das Projekt verfolgt nicht das Ziel, das gesamte Paper oder alle darin enthaltenen Strategien zu replizieren. Repliziert wird nur das **LSTM + CPD Setup** inklusive Feature-Pipeline, Changepoint-Detection-Modul, Trainingslogik und Evaluationsrahmen.

---

## Kurzbeschreibung des Forschungsgegenstands

Das zugrunde liegende Modell kombiniert zwei Bausteine:

1. **Online Changepoint Detection (CPD)** auf Basis von Gaussian Processes, um abrupte Marktveränderungen bzw. Zustandswechsel in Renditeserien zu erkennen.
2. **LSTM-basiertes Deep Momentum Network**, das diese CPD-Signale gemeinsam mit Multi-Horizon-Returns und MACD-Features verarbeitet, um Positionsgrößen direkt zu lernen.

Die Kernidee ist, dass das Modell gleichzeitig

- **langsame Momentum-Effekte** über persistente Trends,
- und **schnelle Mean-Reversion-Effekte** rund um lokale Umbrüche

modellieren kann.

Der Forschungsanspruch des Projekts besteht darin, diese Idee nicht nur konzeptionell nachzubauen, sondern sie in eine **implementierbare, überprüfbare und datengetreue Forschungs-Pipeline** zu überführen.

---

## Projektrahmen

### Scope

Im Scope dieses Projekts sind:

- das LSTM-CPD-Modell aus dem Paper,
- die papernahe Daten- und Featurelogik,
- die Trainings- und Validierungslogik,
- die Replikationsspezifikation,
- die darauf basierende Implementierung,
- sowie die finale dokumentierte Research-Umsetzung im Notebook.

Nicht im Scope sind:

- eine Vollreplikation aller Paper-Benchmarks,
- freie Modellabwandlungen ohne explizite Entscheidung,
- externe oder synthetische Datensätze,
- unsaubere Annahmen an Stellen, an denen das Paper unklar ist.

---

## Datenbasis und Quellenrahmen

Die Datenbasis dieses Projekts ist strikt eingeschränkt auf die projektintern definierten FTMO-Datenquellen.

Verwendet werden:

- `ftmo_assets_nach_kategorie.md` als Definition des vollständigen Asset-Universums,
- `FTMO Data_struktur.md` als Definition der verfügbaren Dateipfade und Zeitrahmen,
- die Paper-Datei *Slow Momentum with Fast Reversion* als Primärquelle für die Modellspezifikation,
- der ausgearbeitete Spec als technische Grundlage für Planung, Task-Zerlegung und Implementierung.

Es dürfen **keine externen oder synthetischen Datenquellen** in die Replikation einfließen.

---

## Methodischer Arbeitsmodus

Das Projekt folgt einem festen 4-Stufen-Workflow:

### 1. Specification
Exakte Extraktion aller für die Replikation relevanten Modell- und Methodenentscheidungen aus dem Paper.

### 2. Plan
Überführung der Spezifikation in einen strukturierten technischen Umsetzungsplan.

### 3. Tasks
Zerlegung des Plans in atomare, eindeutige und implementierbare Arbeitspakete.

### 4. Implementation
Umsetzung ausschließlich auf Basis der zuvor festgezurrten Tasks, ohne zusätzlichen Interpretationsspielraum.

Dieser Aufbau dient dazu, Forschung, Spezifikation und Implementierung strikt voneinander zu trennen und die Reproduzierbarkeit maximal zu erhöhen.

---

## Technische Zielsetzung

Die technische Zielsetzung des Projekts ist die Erstellung einer belastbaren Python-Implementierung des LSTM-CPD-Ansatzes, die:

- auf den projektdefinierten FTMO-Daten läuft,
- die Paperlogik so nah wie möglich abbildet,
- alle offenen Paper-Stellen explizit dokumentiert behandelt,
- reproduzierbar ausgeführt werden kann,
- und am Ende in einem klar strukturierten Jupyter Notebook zusammengeführt wird.

Dieses Notebook ist die finale Forschungsabgabe und soll sowohl die Datenverarbeitung als auch Feature-Bildung, Modelltraining, Evaluation und zentrale Ergebnisse nachvollziehbar abbilden.

---

## Replikationsprinzipien

Für das gesamte Projekt gelten folgende Grundprinzipien:

- **Paper first**: Explizite Aussagen aus dem Paper haben Vorrang.
- **Keine stillen Annahmen**: Unklare Punkte werden markiert und bewusst entschieden.
- **Datenstrenge**: Es werden nur die im Projekt definierten Datenquellen verwendet.
- **Reproduzierbarkeit vor Improvisation**: Jede Entscheidung soll implementierbar, überprüfbar und dokumentierbar sein.
- **Kein Scope Drift**: Modellanpassungen oder Erweiterungen erfolgen nicht implizit.

---

## Erwartetes Endergebnis

Am Ende des Projekts soll eine vollständige, saubere und nachvollziehbare Forschungsartefakt-Sammlung vorliegen, bestehend aus:

- einem finalen **Spec-Dokument**,
- einem strukturierten **Implementierungsplan**,
- einer atomaren **Task-Liste**,
- einer funktionierenden **Python-Implementierung**,
- und einem finalen **Jupyter Notebook**, das die gesamte Replikation dokumentiert.

---

## Projektbeschreibung in einem Satz

Dieses Projekt repliziert das im Paper *Slow Momentum with Fast Reversion* beschriebene **LSTM-CPD-Handelsmodell** auf Basis der projektspezifischen FTMO-Daten, mit dem Ziel einer sauberen, nachvollziehbaren und reproduzierbaren End-to-End-Forschungsumsetzung in Python.


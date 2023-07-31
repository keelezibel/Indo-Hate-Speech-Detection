# Indo Hate Speech Detection

Finetuned BERT model from [here](https://huggingface.co/indobenchmark/indobert-large-p2)

Finetuning Dataset: [id-multi-label-hate-speech-and-abusive-language-detection
](https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection)

## Input:
```
{
    "text": "PKS : Partai Kont*l Sapi... PAN : Partai Anjink Ngent*t, dan Gerindra :Gerakan Intimidasi Rakyat... 3 partai ini adalah partai pengkhianat..."
}
```

## Response:
```
{
    "hs": [
        "false",
        51.83
    ],
    "abusive": [
        "true",
        62.29
    ],
    "hs_individual": [
        "true",
        74.44
    ],
    "hs_group": [
        "true",
        50.42
    ],
    "hs_religion": [
        "true",
        72.1
    ],
    "hs_race": [
        "true",
        54.87
    ],
    "hs_physical": [
        "true",
        54.63
    ],
    "hs_gender": [
        "false",
        58.1
    ],
    "hs_other": [
        "true",
        66.64
    ],
    "hs_weak": [
        "false",
        50.6
    ],
    "hs_moderate": [
        "true",
        62.49
    ],
    "hs_strong": [
        "true",
        50.47
    ]
}
```
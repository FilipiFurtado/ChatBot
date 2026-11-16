# 🍕 Chatbot com GPT-OSS 20B — Prompt Engineering e Atendimento de Pizzaria

Projeto educacional em Python e Jupyter Notebook inspirado no curso [ChatGPT Prompt Engineering for Developers](https://www.deeplearning.ai/courses/chatgpt-prompt-eng/), da DeepLearning.AI, ministrado por Isa Fulford e Andrew Ng.

O projeto começa com experimentos de *prompt engineering* e evolui para um chatbot de atendimento de uma pizzaria fictícia, capaz de conversar em português, consultar um cardápio fornecido no contexto e conduzir a coleta de pedidos.

Em vez de consumir a API hospedada da OpenAI, a aplicação utiliza o modelo de pesos abertos GPT-OSS 20B por meio do Ollama, executado na máquina virtual do Google Colab. As requisições Python são feitas com o SDK `openai`, apontado para a API local do Ollama (`http://localhost:11434/v1`). Não é necessária uma chave de API paga da OpenAI.

Nota: o GPT-OSS 20B foi desenvolvido pela OpenAI, mas não é o mesmo serviço nem o mesmo modelo hospedado no ChatGPT. Neste projeto, “local” significa dentro do ambiente de execução do Colab, e não no computador do usuário.

## Funcionalidades

- Resumir textos: gerar um resumo em uma frase a partir de um trecho delimitado no prompt.
- Seguir instruções de formatação solicitar quebras de linha e outros ajustes na resposta.
- Executar tarefas em etapas: resumir, traduzir para português, formatar e verificar o resultado.
- Classificar mensagens de clientes: identificar sentimento, tipo de mensagem, itens mencionados, endereço e observações; solicitar saída estruturada em JSON.
- Conversar com histórico: utilizar mensagens com papéis system, user e assistant para preservar o contexto entre interações.
- Simular atendimento de pizzaria: apresentar opções do cardápio, perguntar sabores e tamanhos, solicitar informações sobre entrega ou retirada, observações e pagamento.
- Exibir um chat interativo: interface criada com Panel, com mensagens diferenciadas visualmente e área de rolagem.

## Tecnologias utilizadas

| Tecnologia | Finalidade |
| --- | --- |
| Python / Jupyter Notebook | Desenvolvimento e experimentação |
| Google Colab | Ambiente de execução em nuvem |
| NVIDIA Tesla T4 | Aceleração das inferências no ambiente utilizado no projeto |
| GPT-OSS 20B | Modelo de linguagem de pesos abertos |
| Ollama | Download, carregamento e servidor local de inferência |
| OpenAI Python SDK | Comunicação com a API compatível do Ollama |
| Panel | Construção da interface gráfica do chatbot |

## Arquitetura

Usuário
   │
   ▼
Interface Panel (TextInput + botão + histórico visual)
   │
   ▼
collect_messages() ───► context (system / user / assistant)
   │                               │
   └───────────────┬───────────────┘
                   ▼
       get_completion_from_messages()
                   │
                   ▼
           OpenAI Python SDK
                   │
                   ▼
      http://localhost:11434/v1
                   │
                   ▼
             Servidor Ollama
                   │
                   ▼
        GPT-OSS 20B (GPU/CPU)
                   │
                   ▼
        Resposta → histórico e Panel


O `system prompt` define o papel de atendente e inclui informações fictícias da Pizzaria Bella Napoli: horários, entrega, pagamento, regras de montagem do pedido e preços do cardápio. Os próximos turnos são acrescentados à lista `context`, permitindo que o modelo considere o diálogo anterior.

## Pré-requisitos

- Conta Google e acesso ao [Google Colab](https://colab.research.google.com/).
- Ambiente de execução Python 3 com GPU habilitada, preferencialmente uma GPU com memória suficiente para o GPT-OSS 20B.
- Conexão com a internet para instalar pacotes e baixar o modelo (aproximadamente 14 GB, além do espaço para dependências).
- Acesso às células do notebook do projeto ou a uma cópia dos códigos descritos abaixo.

Memória e disponibilidade: o GPT-OSS 20B é recomendado para sistemas com cerca de 16 GB ou mais de memória disponível para o modelo. A T4 pode apresentar aproximadamente 14,6 GiB de VRAM e, portanto, ficar no limite: o Ollama pode usar também a RAM e a CPU, reduzindo o desempenho. A disponibilidade de GPU, os limites de uso e a duração das sessões do Colab não são garantidos, especialmente no plano gratuito.

## Preparação do ambiente no Google Colab

Execute as células na ordem, no mesmo ambiente de execução.

### 1. Habilitar a GPU

No Colab, acesse Ambiente de execução → Alterar o tipo de ambiente de execução → Acelerador de hardware e selecione T4 GPU ou outra GPU disponível. Verifique a configuração:

```python
!nvidia-smi
```

A execução utilizada no desenvolvimento reconheceu uma NVIDIA Tesla T4. O hardware atribuído a outra pessoa pode ser diferente.

### 2. Instalar as dependências do sistema e o Ollama

O instalador do Ollama utiliza `zstd` para descompactar os arquivos distribuídos no formato `.tar.zst`:

```python
!apt-get update -qq
!apt-get install -y zstd curl
!curl -fsSL https://ollama.com/install.sh | sh
```

O aviso `systemd is not running` pode aparecer no Colab: o servidor será iniciado manualmente na próxima etapa.

### 3. Iniciar o servidor Ollama sem bloquear o notebook

Não use `!ollama serve` diretamente em uma célula, porque esse comando permanece ativo esperando requisições. Inicie o processo separadamente e verifique se ele já existe:

```python
import subprocess
import time
import urllib.request

OLLAMA_URL = "http://127.0.0.1:11434/api/version"

def ollama_ativo():
    try:
        with urllib.request.urlopen(OLLAMA_URL, timeout=2) as response:
            return response.status == 200
    except Exception:
        return False

if not ollama_ativo():
    server = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    for _ in range(30):
        if ollama_ativo():
            break
        time.sleep(1)

if not ollama_ativo():
    raise RuntimeError("O servidor Ollama não iniciou corretamente.")

print("Servidor Ollama disponível em http://127.0.0.1:11434")
```

Opcionalmente, confirme a API:

```python
!curl -s http://127.0.0.1:11434/api/version
```

### 4. Baixar o GPT-OSS 20B

```python
!ollama pull gpt-oss:20b
!ollama list
```

O download é feito para o armazenamento do ambiente do Colab. Evite usar `!ollama run gpt-oss:20b` no fluxo do notebook: sem fornecer uma mensagem, o comando abre um chat interativo e pode deixar a célula em execução. Para os exercícios, o modelo será chamado diretamente pelo Python.

### 5. Instalar as bibliotecas Python

```python
%pip install -q openai panel
```

Não é preciso configurar `OPENAI_API_KEY` nem carregar um arquivo `.env` para este projeto. A string `"ollama"` abaixo funciona como valor fictício exigido pelo cliente OpenAI, pois o endpoint está no próprio ambiente Colab.

### 6. Configurar a comunicação Python → Ollama

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    timeout=300.0,
)

def get_completion(prompt, model="gpt-oss:20b"):
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        max_completion_tokens=4096,
        reasoning_effort="low",
        extra_body={
            "keep_alive": "5m",
            "options": {"num_ctx": 4096},
        },
    )

    resposta = ""
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)
            resposta += delta.content

    print()
    return resposta
```

Teste a conexão:

```python
response = get_completion(
    "Resuma em uma frase o conceito de prompt engineering."
)
```

A função imprime os fragmentos em streaming e também retorna uma `str` completa. Para evitar saída duplicada no Jupyter, não coloque `response` isoladamente no fim da mesma célula depois de já imprimir via streaming.

### 7. Executar os exemplos e iniciar a interface

Depois da configuração, execute as células do notebook que implementam:

1. Exercícios de resumo, formatação e instruções em etapas.
2. Exemplos de classificação e extração de informações de mensagens de clientes.
3. A função `get_completion_from_messages(messages)`, que recebe o histórico completo.
4. A interface `Panel`, incluindo `pn.extension()`, `chat_area`, `criar_mensagem()`, `collect_messages()` e `dashboard`.
5. O `context` inicial com as instruções da pizzaria e seu cardápio; por fim, exiba `dashboard`.

Para a interface, o notebook usa um `pn.widgets.TextInput`, um botão associado a `collect_messages` por `on_click()` e uma `pn.Column` com rolagem para exibir o histórico. O histórico de conversação não é automaticamente salvo em disco: ele existe enquanto a sessão de Python estiver ativa.

### 8. Verificar o uso da GPU e liberar recursos

```python
!ollama ps       # Modelos carregados e distribuição GPU/CPU
!nvidia-smi      # Memória da GPU e processos
```

Ao finalizar:

```python
!ollama stop gpt-oss:20b
```

O comando `ollama stop` descarrega o modelo; o servidor pode continuar ativo para receber novas requisições. Encerrar ou reiniciar a sessão do Colab também remove os processos daquela sessão.

## Conceitos estudados

### Prompt engineering

Consiste em elaborar instruções claras e contextualizadas para orientar a saída de um modelo de linguagem. O projeto experimenta instruções diretas, delimitação de texto com marcadores como `####`, formatação e tarefas sequenciais.

### Resumo, transformação e inferência

Os exemplos demonstram três usos de LLMs: resumir um texto, transformar seu conteúdo (por exemplo, traduzi-lo e formatá-lo) e inferir atributos de mensagens, como sentimento e finalidade. Para a classificação, os prompts solicitam chaves como `sentiment`, `kind`, `items`, `address` e `observation` em formato JSON.

### Mensagens de sistema e memória de conversa

A API de chat recebe uma lista de mensagens. A mensagem `system` descreve o comportamento esperado; as mensagens `user` e `assistant` registram o diálogo. A cada turno, `collect_messages()` acrescenta a entrada do usuário e a resposta do modelo a `context`.

### Geração em streaming

Com `stream=True`, a API envia partes da resposta assim que ficam disponíveis. O notebook imprime cada fragmento com `flush=True`, facilitando a observação do resultado mesmo quando a geração leva algum tempo.

### Inferência local e compatibilidade de APIs

Ollama hospeda o modelo na VM do Colab e oferece uma API HTTP. O SDK `openai` se comunica com esse servidor ao mudar o `base_url`, reaproveitando o padrão de requisições estudado no curso. O parâmetro `reasoning_effort="low"` foi utilizado nos experimentos para priorizar respostas mais diretas.

### Interface orientada a eventos

A biblioteca Panel permite construir uma interface simples em Jupyter. O botão de envio executa uma função Python, atualiza o histórico de mensagens e adiciona novos componentes visuais à área de conversa.

## Resultado final: atendente virtual da Pizzaria Bella Napoli

O resultado é um protótipo funcional de chatbot educacional com interface gráfica, executado no Colab. O atendente recebe instruções com o cardápio e as regras da pizzaria fictícia e pode simular o fluxo de atendimento: esclarecer dúvidas, conversar sobre pizzas e bebidas, coletar um pedido, perguntar sobre entrega ou retirada e solicitar a forma de pagamento.

O cardápio incluído no projeto oferece pizzas tradicionais, especiais e doces, além de bebidas, acompanhamentos e combos. Para as pizzas, há preços por tamanho Pequena, Média e Grande; pizzas meio a meio usam, conforme a regra declarada, o preço do sabor mais caro. A taxa fictícia de entrega é de R$ 7,00.

Exemplo ilustrativo de interação (não é uma transcrição de execução do notebook):

> Cliente: Boa noite! Quero uma pizza grande de calabresa e uma grande de quatro queijos. Vocês entregam?
>
> Assistente: Boa noite! Bem-vindo à Pizzaria Bella Napoli! 🍕 Fazemos entregas. A calabresa grande custa R$ 44,90 e a quatro queijos grande, R$ 52,90. O subtotal é R$ 97,80 e a taxa de entrega é R$ 7,00, totalizando R$ 104,80. Qual é o endereço de entrega?
>
> Cliente: Vou enviar o endereço. Posso pagar por Pix?
>
> Assistente: Sim, aceitamos Pix! Pode informar o endereço completo? Depois confirmaremos todos os dados antes de finalizar.

Esse exemplo ilustra o comportamento pretendido pelo prompt. Como as respostas são geradas por um LLM, os resultados efetivos podem variar.

## Limitações e possíveis melhorias

- Protótipo, não sistema comercial: não há integração com banco de dados, estoque, entregas, pagamentos reais ou emissão de pedidos.
- Confiabilidade: o modelo pode interpretar incorretamente preferências, inventar informações ou errar valores. Para uso real, preços, totais e regras de negócio devem ser validados em funções Python, não apenas por texto no prompt.
- Dados fictícios: cardápio, horários, preços e taxas são exemplos didáticos, não informações de um estabelecimento real.
- Estado da aplicação: o histórico é mantido em memória; ao reiniciar o ambiente, a conversa é perdida.
- Recursos do Colab: a sessão pode expirar; o download do modelo e a preparação do ambiente podem precisar ser repetidos.
- Saída JSON: uma instrução pedindo JSON não garante um JSON válido em todos os casos. Uma evolução seria validar o resultado com `json.loads()` e utilizar formatos estruturados quando necessário.

Entre as extensões possíveis estão: cadastro estruturado de pedidos, cálculo automático por item e tamanho, verificação de disponibilidade, persistência de conversas e testes automatizados de intenções e respostas.

## Solução de problemas

| Sintoma | Causa provável / ação recomendada |
| --- | --- |
| `This version requires zstd for extraction` | Instale `zstd` com `apt-get` e repita a instalação do Ollama. |
| `systemd is not running` | Aviso esperado em ambientes como Colab; inicie `ollama serve` com `subprocess.Popen`. |
| A célula com `!ollama serve` não termina | O servidor é um processo contínuo; inicie-o em segundo plano. |
| O download não termina | Confira conectividade, armazenamento disponível e mensagens do `ollama pull`. |
| A resposta demora muito | Verifique `!ollama ps` e `!nvidia-smi`; o modelo pode estar usando CPU além da GPU. |
| Não aparece texto durante a requisição | Use `stream=True` e imprima `delta.content` com `flush=True`. |
| A resposta aparece duas vezes | Não exiba o retorno `response` separadamente se a função já imprimiu o streaming. |
| `AttributeError: ... value_imput` | Use `inp.value` ou `inp.value_input`; `value_imput` não existe. |
| Letras ilegíveis no chat Panel | Defina cores de texto e fundo explicitamente nos componentes visuais. |
| `KeyError: OPENAI_API_KEY` | O endpoint local do Ollama não requer essa variável; configure `api_key="ollama"`. |

## Referências e créditos

- Curso-base: [ChatGPT Prompt Engineering for Developers — DeepLearning.AI](https://www.deeplearning.ai/courses/chatgpt-prompt-eng/)
- Modelo: [Introducing gpt-oss — OpenAI](https://openai.com/index/introducing-gpt-oss/)
- Guia oficial: [How to run gpt-oss locally with Ollama — OpenAI Cookbook](https://github.com/openai/openai-cookbook/blob/main/articles/gpt-oss/run-locally-ollama.md)
- Instalação: [Ollama para Linux](https://ollama.com/download/linux)
- Integração: [Ollama — OpenAI compatibility](https://github.com/ollama/ollama/blob/main/docs/api/openai-compatibility.mdx)
- Ambiente: [Google Colab — FAQ](https://research.google.com/colaboratory/intl/pt-BR/faq.html)
- Interface: [Panel](https://panel.holoviz.org/)

Projeto desenvolvido para fins educacionais, com base nos conceitos apresentados por Isa Fulford (OpenAI) e Andrew Ng (DeepLearning.AI). O chatbot da pizzaria é uma adaptação prática, independente do curso original.

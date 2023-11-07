# SAVI-TP1

Pratical Work 1  - Real Time Face Recognition script <br>
SAVI - 2023/2024 Universidade de Aveiro
<br>
<br>
## Contribuidores: 
### Grupo 5
- José Silva (103268) josesilva8@ua.pt
- Mário Vasconcelos (84081) mario.vasconcelos@ua.pt
- Nuno Cunha (95167) nunocunha99@ua.pt
 <br>
 
## Instalação
Dependências de pacotes:
- `pip install opencv-python`
- `pip install mediapipe`
- `pip install face_recognition`
- `pip install dlib==19.22`
- `pip install gtts`
- `pip install pygame`
<br>

## Guia de utilizador
 Este programa conta com dois modos de funcionamento:
 - Modo "cdm", "Collect Data Mode", dedicado à recolha de imagens de pessoas para armazenar numa base de dados.
 - Modo "fdm", "Face Detection Mode", que executa o reconhecimento facial da pessoa que aparece na câmara e compara a imagem atual com as imagens presentes na base de dados.

 Abordando primeiramente o modo "cdm", este consiste numa função onde são detetados rostos que vão aparecendo na câmara, neste caso, na webcam. Após a deteção, o utilizador tira as fotos que achar necessárias para um bom processamento de reconhecimento facial. Pressionando 'f' ou 'l', através do teclado, pode tirar fotos frontais e de perfil, estas são guardadas como "nome_frontal.jpg" e "nome_profile.jpg". Para além destas "presskeys", o utilizador também tem a possibilidade de, ao carregar em 'p', pausar a stream da câmara  e ao pressionar 'q' abandonar o programa e fechar todas as janelas.

 Referente ao modo "fdm", é feito o processo de reconhecimento facial e tracking da cara dos indivíduos que aparecem na stream. Este programa tem a capacidade de importar todas as fotos existentes na base de dados para poder ser realizada uma comparação com a imagem atual (imagem da stream), determinando assim deste modo se o indíviduo que aparece no vídeo é conhecido ou desconhecido. Tem também a possibilidade de iniciar o programa sem imagens na base de dados, permitindo adiconar pessoas novas à base de dados no decorrer do programa e por fim, também tem a capacidade de fazer tracking da caras que aparecem no vídeo. No decorrer do programa é possível utilizar as seguintes presskeys:
 - 'p' para pausar o vídeo;
 - 'q' para sair do programa e fechar todas as janelas;
 - 'd' para apagar a base de dados de imagens e áudios;
 - 'r' para o programa passar a reconhecer pessoas desconhecidas.
<br>

## Demonstração
O video demonstra um funcionamento rápido da aplicação

![Video Exemplo](https://github.com/Nunoc99/SAVI-TP1/assets/145439915/30aab01c-499d-4956-ba0b-84843fd5426f)





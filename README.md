from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Union
import numpy as np
import yaml

N_FILAS = 3
N_COLS = 3

class JugadorTresEnRaya(ABC):
    def __init__(self, nombre: str) -> None:
        """
        Inicializa un jugador con un nombre.

        :param nombre: Nombre del jugador.
        """
        self.name = nombre
        self._estados_juego_actual = []  # Conjunto de estados visitados en la partida en curso

    @abstractmethod  # Las clases hijas deben implementar este método
    def decide_accion(
        self, posiciones: List[Tuple[int, int]], tablero: np.ndarray, token: int
    ) -> Tuple[int, int]:
        """
        Dado un tablero y un token, decide la próxima acción a realizar.

        :param posiciones: Lista de posiciones libres en el tablero.
        :param tablero: Estado actual del tablero.
        :param token: Token del jugador actual.
        :return: Tupla con las coordenadas de la acción a realizar.
        """
        pass

    def reset(self):
        """
        Resetea el jugador para una nueva partida.
        """
        self._estados_juego_actual = []  # Resetea la lista de estados visitados en la partida en curso

    def guarda_politica(self, path_artefacto: Path):
        """
        Guarda la política del jugador en un fichero YAML.

        :param path_artefacto: Ruta donde se guardará el fichero YAML.
        """
        with open(path_artefacto, "w") as f:
            yaml.dump(self._experiencia_estado_valor, f, Dumper=yaml.Dumper)

    def carga_politica(self, path_artefacto: Path):
        """
        Carga la política de un fichero YAML.

        :param path_artefacto: Ruta del fichero Pickle a YAML.
        """
        with open(path_artefacto, "rb") as f:
            self._experiencia_estado_valor = yaml.load(f, Loader=yaml.FullLoader)


class JugadorTresEnRayaMaq(JugadorTresEnRaya):
    def __init__(
        self,
        nombre,
        tasa_exploracion: float = 0.3,
        tasa_aprendizaje: float = 0.2,
        descuento_gamma: float = 0.9,
    ):
        """
        Inicializa un jugador máquina.

        :param nombre: Nombre del jugador.
        :param tasa_exploracion: Ratio de exploración para los jugadores máquina. Por defecto, 0.3. Indica la probabilidad de que el jugador
            elija una acción aleatoria en lugar de la mejor acción posible. No tiene efecto en los jugadores humanos.
        :param tasa_aprendizaje: Tasa de aprendizaje para el jugador.
        :param descuento_gamma: Factor de descuento gamma para el jugador.
        """
        super().__init__(nombre)
        self._tasa_exploracion = tasa_exploracion
        self._tasa_aprendizaje = tasa_aprendizaje
        self._descuento_gamma = descuento_gamma
        self._experiencia_estado_valor = {}

    def decide_accion(
        self, posiciones: List[Tuple[int, int]], tablero: np.ndarray, token: int
    ) -> Tuple[int, int]:
        """
        Lógica de la máquina para decidir la acción. En este caso, se elige la acción aleatoria o la mejor según un algoritmo.
        """
        if np.random.rand() < self._tasa_exploracion:
            # Acción aleatoria
            return posiciones[np.random.choice(len(posiciones))]
        else:
            # Lógica para elegir la mejor acción basada en la política
            # Aquí va la lógica para seleccionar la mejor acción (por ejemplo, usando Q-learning)
            return posiciones[0]  # Selección dummy para el ejemplo

    def guarda_estado(self, s):
        """
        Guarda el estado actual del jugador en la lista de estados visitados de la partida.

        :param estado: Hash del estado actual del tablero.
        """
        self._estados_juego_actual.append(s)

    def retropropaga_recompensa(self, recompensa_final: float) -> None:
        """
        Retropropaga la recompensa final de la partida a la serie de estados visitados por el jugador, para
        aprender de la partida.

        :param recompensa: Recompensa final de la partida.
        """
        # Aquí iría la retropropagación, actualizando la política con la recompensa
        pass


class JugadorTresEnRayaHum(JugadorTresEnRaya):
    def decide_accion(
        self, posiciones: List[Tuple[int, int]], tablero: np.ndarray, token: int
    ) -> Tuple[int, int]:
        print(f"Posiciones libres: {posiciones}")
        while True:
            try:
                x, y = (
                    input(
                        "Introduzca la posición (`fila, columna`) donde desea jugar: "
                    )
                    .strip()
                    .split(",")
                )
                x = int(x)
                y = int(y)
                if (x, y) in posiciones:
                    return x, y
                else:
                    raise ValueError("Posición no válida.")
            except:
                print("Entrada no válida. Inténtelo de nuevo...")
                continue


class JuegoTresEnRaya:
    def __init__(
        self, jugador1: JugadorTresEnRayaMaq, jugador2: JugadorTresEnRaya
    ) -> None:
        """
        Inicializa el juego con dos jugadores. El jugador 1 siempre será una máquina, y el jugador 2 puede ser
        una máquina o un humano.
        """
        self._jugador1 = jugador1
        self._jugador2 = jugador2
        self._n_filas = N_FILAS
        self._n_cols = N_COLS
        self._tablero = np.zeros((self._n_filas, self._n_cols))
        self._fin = False
        self._estado = None
        self._siguiente_jugador = 1  # Empieza el jugador 1, luego se cambia a -1, jugador 2

    @staticmethod
    def _serializa_estado(tablero: np.ndarray, n_filas: int, n_cols: int) -> str:
        """
        Calcula una representación en string del tablero actual, para poder ser guardado en un diccionario.
        """
        return str(tablero.reshape(n_cols * n_filas))

    def __calcula_ganador(self) -> Union[int, None]:
        """
        Implementa la lógica para determinar si hay un ganador en el tablero actual.
        """
        # Fila
        for i in range(self._n_filas):
            if sum(self._tablero[i, :]) == 3:
                self._fin = True
                return 1
            if sum(self._tablero[i, :]) == -3:
                self._fin = True
                return -1

        # Columna
        for i in range(self._n_cols):
            if sum(self._tablero[:, i]) == 3:
                self._fin = True
                return 1
            if sum(self._tablero[:, i]) == -3:
                self._fin = True
                return -1

        # Diagonal
        diag_sum1 = sum([self._tablero[i, i] for i in range(self._n_cols)])
        diag_sum2 = sum(
            [self._tablero[i, self._n_cols - i - 1] for i in range(self._n_cols)]
        )
        diag_sum = max(abs(diag_sum1), abs(diag_sum2))
        if diag_sum == 3:
            self._fin = True
            if diag_sum1 == 3 or diag_sum2 == 3:
                return 1
            else:
                return -1

        # Empate
        if (
            len(self.__get_posiciones_libres()) == 0
        ):  # Si no quedan posiciones disponibles
            self._fin = True
            return 0

        # Sin finalizar
        self._fin = False
        return None

    def __get_posiciones_libres(self) -> List[Tuple[int, int]]:
        """
        Devuelve una lista con las posiciones libres en el tablero.
        """
        positions = []
        for i in range(self._n_filas):
            for j in range(self._n_cols):
                if self._tablero[i, j] == 0:
                    positions.append((i, j))
        return positions

    def __actualiza_estado(self, position: Tuple[int, int]) -> None:
        """
        Actualiza el estado del tablero tras una jugada.
        """
        self._tablero[position] = self._siguiente_jugador
        self._siguiente_jugador = (
            -1 if self._siguiente_jugador == 1 else 1
        )  # Cambia de jugador
        self._estado = self._serializa_estado(
            self._tablero, self._n_filas, self._n_cols
        )

    def print_tablero(self, verboso: bool = True) -> None:
        """
        Imprime el tablero actual por pantalla.
        """
        if verboso:
            for i in range(0, self._n_filas):
                print(" | ".join([str(int(x)) for x in self._tablero[i]]))
                if i < self._n_filas - 1:
                    print("-" * 9)

    def play(self) -> None:
        """
        Juega el juego hasta que haya un ganador.
        """
        while not self._fin:
            if self._siguiente_jugador == 1:
                accion = self._jugador1.decide_accion(
                    self.__get_posiciones_libres(), self._tablero, 1
                )
            else:
                accion = self._jugador2.decide_accion(
                    self.__get_posiciones_libres(), self._tablero, -1
                )
            self.__actualiza_estado(accion)
            self.print_tablero()
            resultado = self.__calcula_ganador()
            if resultado is not None:
                if resultado == 0:
                    print("Empate!")
                elif resultado == 1:
                    print("¡Gana Jugador 1!")
                elif resultado == -1:
                    print("¡Gana Jugador 2!")
                break

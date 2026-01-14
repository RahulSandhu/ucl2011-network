import os

# Paths
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "../data/")
IMAGES_DIR = os.path.join(BASE_DIR, "../images/")
RESULTS_DIR = os.path.join(BASE_DIR, "../results/")

# Player name mappings
PLAYER_NAME_MAP = {
    # FC Barcelona
    "Víctor Valdés Arribas": "Valdés",
    "Daniel Alves da Silva": "Dani Alves",
    "Gerard Piqué Bernabéu": "Piqué",
    "Javier Alejandro Mascherano": "Mascherano",
    "Eric-Sylvain Bilal Abidal": "Abidal",
    "Sergio Busquets i Burgos": "Busquets",
    "Xavier Hernández Creus": "Xavi",
    "Andrés Iniesta Luján": "Iniesta",
    "Pedro Eliezer Rodríguez Ledesma": "Pedro",
    "Lionel Andrés Messi Cuccittini": "Messi",
    "David Villa Sánchez": "Villa",
    "Carles Puyol i Saforcada": "Puyol",
    "Seydou Kéita": "Keita",
    "Ibrahim Afellay": "Afellay",
    # Manchester United
    "Edwin van der Sar": "Van der Sar",
    "Fábio Pereira da Silva": "Fábio",
    "Rio Ferdinand": "Ferdinand",
    "Nemanja Vidić": "Vidić",
    "Patrice Evra": "Evra",
    "Antonio Valencia": "Valencia",
    "Michael Carrick": "Carrick",
    "Ryan Giggs": "Giggs",
    "Ji-Sung Park": "Park",
    "Wayne Mark Rooney": "Rooney",
    "Javier Hernández Balcázar": "Chicharito",
    "Paul Scholes": "Scholes",
    "Luis Antonio Valencia Mosquera": "Valencia",
    "Luís Carlos Almeida da Cunha": "Nani",
}

# Player image paths
PLAYER_IMAGE_PATHS = {
    # FC Barcelona
    "Valdés": os.path.join(IMAGES_DIR, "fcb/victor_valdes.jpg"),
    "Dani Alves": os.path.join(IMAGES_DIR, "fcb/dani_alves.jpg"),
    "Piqué": os.path.join(IMAGES_DIR, "fcb/gerard_pique.jpg"),
    "Mascherano": os.path.join(IMAGES_DIR, "fcb/javier_mascherano.jpg"),
    "Abidal": os.path.join(IMAGES_DIR, "fcb/eric_abidal.jpg"),
    "Busquets": os.path.join(IMAGES_DIR, "fcb/sergio_busquets.jpg"),
    "Xavi": os.path.join(IMAGES_DIR, "fcb/xavi.jpg"),
    "Iniesta": os.path.join(IMAGES_DIR, "fcb/andres_iniesta.jpg"),
    "Pedro": os.path.join(IMAGES_DIR, "fcb/pedro.jpg"),
    "Messi": os.path.join(IMAGES_DIR, "fcb/lionel_messi.jpg"),
    "Villa": os.path.join(IMAGES_DIR, "fcb/david_villa.jpg"),
    "Puyol": os.path.join(IMAGES_DIR, "fcb/carles_puyol.jpg"),
    "Keita": os.path.join(IMAGES_DIR, "fcb/seydou_keita.jpg"),
    "Afellay": os.path.join(IMAGES_DIR, "fcb/ibrahim_afellay.jpg"),
    # Manchester United
    "Van der Sar": os.path.join(IMAGES_DIR, "man_utd/edwin_van_der_sar.jpg"),
    "Fábio": os.path.join(IMAGES_DIR, "man_utd/fabio.jpg"),
    "Ferdinand": os.path.join(IMAGES_DIR, "man_utd/rio_ferdinand.jpg"),
    "Vidić": os.path.join(IMAGES_DIR, "man_utd/nemanja_vidic.jpg"),
    "Evra": os.path.join(IMAGES_DIR, "man_utd/patrice_evra.jpg"),
    "Valencia": os.path.join(IMAGES_DIR, "man_utd/antonio_valencia.jpg"),
    "Carrick": os.path.join(IMAGES_DIR, "man_utd/michael_carrick.jpg"),
    "Giggs": os.path.join(IMAGES_DIR, "man_utd/ryan_giggs.jpg"),
    "Park": os.path.join(IMAGES_DIR, "man_utd/jisung_park.jpg"),
    "Rooney": os.path.join(IMAGES_DIR, "man_utd/wayne_rooney.jpg"),
    "Chicharito": os.path.join(IMAGES_DIR, "man_utd/chicharito.jpg"),
    "Scholes": os.path.join(IMAGES_DIR, "man_utd/paul_scholes.jpg"),
    "Nani": os.path.join(IMAGES_DIR, "man_utd/nani.jpg"),
}

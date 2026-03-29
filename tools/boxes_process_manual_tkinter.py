import os
import json
import tkinter as tk
from tkinter import Tk, Canvas, PhotoImage, simpledialog
from PIL import Image, ImageTk

class ImageManager:
    def __init__(self, root_folder):
        self.root_folder = root_folder
        self.save_folder = 'save'
        self.image_list = []
        self.current_image_index = 0
        self.zoom_factor = 1.0
        self.zoom_step = 1.01
        self.drag_step = 1
        self.cursor_size = 256

        self.setup_folders()
        self.load_images()

    def setup_folders(self):
        os.makedirs(os.path.join(self.root_folder, self.save_folder), exist_ok=True)

    def load_images(self):
        self.image_list = sorted([
            f for f in os.listdir(self.root_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'))
        ])
    
    def reload_images(self):
        """Recharge la liste des images (utile après sauvegarde)"""
        old_count = len(self.image_list)
        self.load_images()
        new_count = len(self.image_list)
        if old_count != new_count:
            print(f"Liste d'images mise à jour: {old_count} -> {new_count} images")

    def get_image_path(self):
        return os.path.join(self.root_folder, self.image_list[self.current_image_index])

    def get_bounding_boxes(self):
        base_name = os.path.splitext(self.image_list[self.current_image_index])[0]
        bbox_files = [f for f in os.listdir(self.root_folder) if f.startswith(base_name) and f.endswith('.json')]

        bounding_boxes = []
        for bbox_file in bbox_files:
            with open(os.path.join(self.root_folder, bbox_file), 'r') as f:
                data = json.load(f)
                # Retourner aussi le nom du fichier JSON comme identifiant unique
                bounding_boxes.append((data['annotation']['bbox'], data['annotation']['category_name'], bbox_file))

        return bounding_boxes

class PhotoViewer:
    def __init__(self, root_folder, category_name, crop_height=0, auto_crop=False):
        self.root = Tk()
        self.root.title("Photo Viewer")
        self.root.geometry("1920x1080")

        self.image_manager = ImageManager(root_folder)
        
        # Créer un frame pour contenir le canvas et la barre de couleur
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill="both", expand=True)
        
        # Canvas pour l'image (en premier pour éviter le décalage)
        self.canvas = Canvas(self.main_frame, cursor="cross", bg="white", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        
        # Barre de couleur en bas (hauteur fixe)
        self.color_bar = tk.Frame(self.main_frame, height=30)
        self.color_bar.pack(fill="x", side="bottom")
        self.color_bar.pack_propagate(False)  # Empêcher le frame de se réduire
        
        self.category_name = category_name
        self.crop_height = crop_height  # Hauteur en pixels à croper (0 = désactivé)
        self.auto_crop = auto_crop  # Croper automatiquement lors de la sauvegarde
        
        # Variables pour l'auto-traitement
        self.auto_processing = False  # Mode auto-traitement actif
        self.auto_process_thread = None  # Thread pour l'auto-traitement

        self.drag_start_x = 0
        self.drag_start_y = 0
        self.selected_bbox = None
        self.selected_handle = None
        self.bbox_handles = []
        self.bboxes = []
        self.bbox_coords = []
        self.bbox_files = []  # Liste des noms de fichiers JSON pour chaque box
        self.bbox_labels = []  # Liste des IDs des labels de texte pour chaque box
        self.deleted_bboxes = set()  # Maintenant stocke les noms de fichiers JSON
        self.highlighted_bbox = None
        self.dragging_bbox = None
        self.drag_offset_x = 0
        self.drag_offset_y = 0

        self.tk_image = None
        self.original_image = None
        self.image = None

        self.bind_events()
        self.load_image()
        self.draw_bounding_boxes()
        self.draw_crop_zone()  # Afficher la zone de crop si activée
        self.update_window_color_from_current_image()  # Mettre à jour la couleur au démarrage

        self.root.mainloop()

    def validate_and_fix_bbox_coordinates(self, data, image_width, image_height):
        """
        Valide et corrige les coordonnées des boîtes pour s'assurer qu'elles sont dans les limites de l'image.
        Retourne les données corrigées.
        """
        if 'annotation' in data and 'bbox' in data['annotation']:
            bbox = data['annotation']['bbox']
            if len(bbox) == 4:
                x, y, w, h = bbox
                
                # Corriger les coordonnées négatives
                x = max(0, x)
                y = max(0, y)
                
                # Corriger les dimensions négatives
                w = max(1, w)  # Largeur minimale de 1 pixel
                h = max(1, h)  # Hauteur minimale de 1 pixel
                
                # S'assurer que la boîte ne dépasse pas les limites de l'image
                if x + w > image_width:
                    w = max(1, image_width - x)
                if y + h > image_height:
                    h = max(1, image_height - y)
                
                # Mettre à jour les coordonnées corrigées
                data['annotation']['bbox'] = [x, y, w, h]
                
                print(f"[✓] Coordonnées corrigées : [{x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}]")
        
        return data

    def ensure_json_consistency(self, data, image_name, image_width, image_height):
        """
        Assure la cohérence du JSON en ajoutant les champs manquants et en validant les coordonnées.
        """
        # S'assurer que la section 'image' existe et est complète
        if 'image' not in data:
            data['image'] = {}
        
        data['image']['file_name'] = image_name
        data['image']['width'] = image_width
        data['image']['height'] = image_height
        
        # S'assurer que la section 'annotation' existe et est complète
        if 'annotation' not in data:
            data['annotation'] = {}
        
        data['annotation']['file_name'] = image_name
        
        # Ajouter bbox_id s'il n'existe pas
        if 'bbox_id' not in data['annotation']:
            # Essayer d'extraire l'ID du nom de fichier
            try:
                bbox_id = int(data.get('annotation', {}).get('bbox_id', 0))
            except (ValueError, TypeError):
                bbox_id = 0
            data['annotation']['bbox_id'] = bbox_id
        
        # Valider et corriger les coordonnées des boîtes
        data = self.validate_and_fix_bbox_coordinates(data, image_width, image_height)
        
        return data

    def save_json_with_consistency_check(self, file_path, data):
        """
        Sauvegarde un JSON avec vérification de cohérence et formatage correct.
        """
        # Obtenir les informations de l'image courante
        current_image_name = self.image_manager.image_list[self.image_manager.current_image_index]
        image_width, image_height = self.original_image.size
        
        # Assurer la cohérence du JSON
        data = self.ensure_json_consistency(data, current_image_name, image_width, image_height)
        
        # Sauvegarder avec formatage correct
        json_string = json.dumps(data, indent=4, ensure_ascii=False, separators=(',', ': '))
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(json_string)
            f.write('\n')  # Ajouter un retour à la ligne final

    def bind_events(self):
        #self.root.bind("<KeyRelease-n>", self.show_next_image)
        self.root.bind("<Button-4>", self.zoom_in)
        self.root.bind("<Button-5>", self.zoom_out)
        self.root.bind("<B1-Motion>", self.on_drag)
        self.root.bind("<Button-1>", self.on_click)
        self.root.bind("<Motion>", self.on_mouse_move)
        self.root.bind("<Button-3>", self.on_right_click)
        self.root.bind("<B3-Motion>", self.on_right_drag)
        self.root.bind("<ButtonRelease-3>", self.on_right_release)
        self.root.bind("<Escape>", lambda event: self.quit_app())
        self.root.bind("<KeyRelease>", self.handle_key_press)
        self.root.bind("<Control-a>", self.start_auto_processing)  # CTRL+a pour démarrer l'auto-traitement

    def handle_key_press(self, event):
        if event.keysym.startswith('KP_'):
            self.handle_numeric_keypad(event.keysym)
        else:
            self.handle_regular_key(event.keysym)

    def handle_numeric_keypad(self, key):
        zoom_values = {
            'KP_Insert': 1.0,
            'KP_End': 0.1,
            'KP_Down': 0.2,
            'KP_Next': 0.3,
            'KP_Left': 0.4,
            'KP_Begin': 0.5,
            'KP_Right': 0.6,
            'KP_Home': 0.7,
            'KP_Up': 0.8,
            'KP_Prior': 0.9
        }
        if key in zoom_values:
            self.image_manager.zoom_factor = zoom_values[key]
            self.zoom()

    def handle_regular_key(self, key):
        actions = {
            'Delete': lambda: self.delete_and_next(),
            'Right': lambda: self.show_next_image(None),
            'Left': lambda: self.show_next_image(None, reverse=True),
            's': self.image_save_folder,
            't': self.image_save_tmp,  # Ajout de l'action 't'
            'd': self.delete_bbox,
            'r': self.edit_category_name,
            'f': self.fill_box_full_image,  # Ajout de l'action 'f'
            'l': self.fill_box_full_width,  # Ajout de l'action 'l'
            'n': self.add_new_box,  # Ajout de l'action 'n'
            'x': self.crop_bottom_zone  # Ajout de l'action 'x' pour croper
        }
        if key in actions:
            actions[key]()

    def add_new_box(self):
        # Crée une nouvelle boxe en (0,0,50,50) avec une catégorie par défaut
        base_name = os.path.splitext(self.image_manager.image_list[self.image_manager.current_image_index])[0]
        # Chercher le prochain numéro disponible
        existing = [f for f in os.listdir(self.image_manager.root_folder) if f.startswith(base_name) and f.endswith('.json')]
        nums = []
        for f in existing:
            try:
                num = int(f.split('_')[-1].split('.')[0])
                nums.append(num)
            except Exception:
                pass
        next_num = max(nums) + 1 if nums else 0
        new_json_name = f"{base_name}_{next_num}.json"
        new_json_path = os.path.join(self.image_manager.root_folder, new_json_name)
        
        # Obtenir les informations de l'image courante
        current_image_name = self.image_manager.image_list[self.image_manager.current_image_index]
        image_width, image_height = self.original_image.size
        
        # Structure du fichier JSON complet et bien formaté
        data = {
            "image": {
                "file_name": current_image_name,
                "width": image_width,
                "height": image_height
            },
            "annotation": {
                "file_name": current_image_name,
                "bbox": [0, 0, 50, 50],
                "category_name": self.category_name,
                "bbox_id": next_num
            }
        }
        
        # Sauvegarder avec contrôle de cohérence
        self.save_json_with_consistency_check(new_json_path, data)
        # Redessiner les boxes (couleur mise à jour automatiquement)
        self.draw_bounding_boxes()

    def delete_and_next(self):
        image_path = self.image_manager.get_image_path()
        base_name = os.path.splitext(self.image_manager.image_list[self.image_manager.current_image_index])[0]

        os.remove(image_path)

        bbox_files = [f for f in os.listdir(self.image_manager.root_folder) if f.startswith(base_name) and f.endswith('.json')]
        for bbox_file in bbox_files:
            os.remove(os.path.join(self.image_manager.root_folder, bbox_file))

        self.show_next_image(None)

    def load_image(self):
        image_path = self.image_manager.get_image_path()
        self.original_image = Image.open(image_path)
        self.image = self.original_image.copy()
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.config(width=self.image.width, height=self.image.height)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        self.root.title(f"{image_path} (Zoom: {round(self.image_manager.zoom_factor, 3)})")

    def show_next_image(self, event, reverse=False):
        if reverse:
            self.image_manager.current_image_index = (
                self.image_manager.current_image_index - 1
            ) % len(self.image_manager.image_list)
        else:
            self.image_manager.current_image_index = (
                self.image_manager.current_image_index + 1
            ) % len(self.image_manager.image_list)

        self.drag_start_x = 0
        self.drag_start_y = 0
        self.image_manager.zoom_factor = 1
        self.deleted_bboxes = set()
        self.load_image()
        
        # Mettre à jour le titre avec la progression
        self.update_title_with_progress()
        self.draw_bounding_boxes()
        self.draw_crop_zone()
        self.update_window_color_from_current_image()  # Mettre à jour la couleur lors du changement d'image

    def zoom_in(self, event):
        self.image_manager.zoom_factor *= self.image_manager.zoom_step
        self.zoom()

    def zoom_out(self, event):
        self.image_manager.zoom_factor /= self.image_manager.zoom_step
        self.zoom()

    def zoom(self):
        little_rang = 0.98
        if self.original_image.height > self.original_image.width:
            zoom_factor_max = self.image_manager.cursor_size / (self.original_image.width * little_rang)
        else:
            zoom_factor_max = self.image_manager.cursor_size / (self.original_image.height * little_rang)

        if self.image_manager.zoom_factor < zoom_factor_max:
            self.image_manager.zoom_factor = zoom_factor_max

        new_width = int(self.original_image.width * self.image_manager.zoom_factor)
        new_height = int(self.original_image.height * self.image_manager.zoom_factor)
        self.image = self.original_image.resize((new_width, new_height))
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.config(width=new_width, height=new_height)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        self.root.title(f"{self.image_manager.get_image_path()} (Zoom: {round(self.image_manager.zoom_factor, 3)})")
        self.draw_bounding_boxes()
        self.draw_crop_zone()

    def on_click(self, event):
        self.drag_start_x = event.x
        self.drag_start_y = event.y

        clicked_item = self.canvas.find_closest(event.x, event.y)[0]

        if clicked_item in self.bbox_handles:
            self.selected_handle = clicked_item
            handle_index = self.bbox_handles.index(clicked_item)
            self.selected_bbox = handle_index // 4  # 4 poignées par boxe maintenant
        else:
            self.selected_handle = None
            self.selected_bbox = None

    def on_drag(self, event):
        if self.selected_handle is not None:
            delta_x = event.x - self.drag_start_x
            delta_y = event.y - self.drag_start_y

            handle_index = self.bbox_handles.index(self.selected_handle)
            bbox_index = self.selected_bbox
            x1, y1, x2, y2 = self.bbox_coords[bbox_index]

            # 4 poignées par boxe : 0=top-left, 1=top-right, 2=bottom-right, 3=bottom-left
            handle_in_box = handle_index % 4
            if handle_in_box == 0:  # Top-left handle
                x1 += delta_x
                y1 += delta_y
            elif handle_in_box == 1:  # Top-right handle
                x2 += delta_x
                y1 += delta_y
            elif handle_in_box == 2:  # Bottom-right handle
                x2 += delta_x
                y2 += delta_y
            else:  # Bottom-left handle (handle_in_box == 3)
                x1 += delta_x
                y2 += delta_y

            self.canvas.coords(self.bboxes[bbox_index], x1, y1, x2, y2)
            
            # Mettre à jour la position du label de catégorie
            if bbox_index < len(self.bbox_labels):
                self.canvas.coords(self.bbox_labels[bbox_index], x1 + 10, y1 - 10)
            
            self.update_bbox_coords(bbox_index, x1, y1, x2, y2)

            self.drag_start_x = event.x
            self.drag_start_y = event.y

    def update_bbox_coords(self, bbox_index, x1, y1, x2, y2):
        self.bbox_coords[bbox_index] = (x1, y1, x2, y2)
        self.update_handles(bbox_index)

    def update_handles(self, bbox_index):
        handle_size = 10
        x1, y1, x2, y2 = self.bbox_coords[bbox_index]

        # 4 poignées par boxe : top-left, top-right, bottom-right, bottom-left
        self.canvas.coords(self.bbox_handles[4 * bbox_index],  # Top-left
                           x1 - handle_size / 2, y1 - handle_size / 2,
                           x1 + handle_size / 2, y1 + handle_size / 2)

        self.canvas.coords(self.bbox_handles[4 * bbox_index + 1],  # Top-right
                           x2 - handle_size / 2, y1 - handle_size / 2,
                           x2 + handle_size / 2, y1 + handle_size / 2)

        self.canvas.coords(self.bbox_handles[4 * bbox_index + 2],  # Bottom-right
                           x2 - handle_size / 2, y2 - handle_size / 2,
                           x2 + handle_size / 2, y2 + handle_size / 2)

        self.canvas.coords(self.bbox_handles[4 * bbox_index + 3],  # Bottom-left
                           x1 - handle_size / 2, y2 - handle_size / 2,
                           x1 + handle_size / 2, y2 + handle_size / 2)

    def get_color_for_bbox_count(self, bbox_count):
        """
        Retourne une couleur basée sur le nombre de bounding boxes.
        0 boxes = gris, 1 = vert, 2+ = rouge (de plus en plus foncé)
        """
        if bbox_count == 0:
            return "#808080"  # Gris
        elif bbox_count == 1:
            return "#00FF00"  # Vert
        else:
            # Rouge de plus en plus foncé (max 5 boxes pour éviter le noir complet)
            intensity = min(bbox_count, 5)
            red_value = min(255, 100 + intensity * 30)
            return f"#{int(red_value):02x}0000"  # Rouge avec intensité variable

    def update_window_color(self, bbox_count):
        """Met à jour la couleur de la barre de couleur et de la barre de titre"""
        color = self.get_color_for_bbox_count(bbox_count)
        
        # Changer la couleur de la barre de couleur
        self.color_bar.configure(bg=color)
        
        # Mettre à jour le titre avec la couleur
        current_title = self.root.title()
        if " | " in current_title:
            base_title = current_title.split(" | ")[0]
        else:
            base_title = current_title
        
        new_title = f"{base_title} | Boxes: {bbox_count} | Color: {color}"
        self.root.title(new_title)

    def update_window_color_from_current_image(self):
        """Met à jour la couleur de la fenêtre basée sur l'image actuelle"""
        # Compter le nombre de boxes (non supprimées) pour l'image actuelle
        all_boxes = self.image_manager.get_bounding_boxes()
        active_boxes = [box for box in all_boxes if box[2] not in self.deleted_bboxes]
        bbox_count = len(active_boxes)
        
        # Mettre à jour la couleur de la fenêtre
        self.update_window_color(bbox_count)

    def draw_bounding_boxes(self):
        self.canvas.delete("bbox")
        self.canvas.delete("crop_zone")  # Nettoyer aussi la zone de crop
        self.bboxes = []
        self.bbox_handles = []
        self.bbox_coords = []
        self.bbox_files = []  # Réinitialiser la liste des fichiers JSON
        self.bbox_labels = []  # Réinitialiser la liste des labels

        # Compter le nombre de boxes (non supprimées)
        all_boxes = self.image_manager.get_bounding_boxes()
        active_boxes = [box for box in all_boxes if box[2] not in self.deleted_bboxes]
        bbox_count = len(active_boxes)
        
        # Mettre à jour la couleur de la fenêtre
        self.update_window_color(bbox_count)

        for idx, (bbox, category_name, bbox_file) in enumerate(all_boxes):
            color = "red" if category_name.lower() == "unknown" else "green"
            if bbox_file not in self.deleted_bboxes:
                x1, y1, width, height = bbox
                x1_zoomed = x1 * self.image_manager.zoom_factor
                y1_zoomed = y1 * self.image_manager.zoom_factor
                x2_zoomed = (x1 + width) * self.image_manager.zoom_factor
                y2_zoomed = (y1 + height) * self.image_manager.zoom_factor

                bbox_id = self.canvas.create_rectangle(x1_zoomed, y1_zoomed, x2_zoomed, y2_zoomed, outline=color, width=2, tags="bbox")
                self.bboxes.append(bbox_id)
                self.bbox_coords.append((x1_zoomed, y1_zoomed, x2_zoomed, y2_zoomed))
                self.bbox_files.append(bbox_file)

                label_id = self.canvas.create_text(x1_zoomed + 10, y1_zoomed - 10, text=category_name, fill=color, font=("Arial", 12), tags="bbox", anchor="nw")
                self.bbox_labels.append(label_id)

                handle_size = 10
                # 4 poignées par boxe : top-left, top-right, bottom-right, bottom-left
                handle1 = self.canvas.create_rectangle(x1_zoomed - handle_size / 2, y1_zoomed - handle_size / 2,
                                                       x1_zoomed + handle_size / 2, y1_zoomed + handle_size / 2,
                                                       fill="blue", tags="bbox")
                handle2 = self.canvas.create_rectangle(x2_zoomed - handle_size / 2, y1_zoomed - handle_size / 2,
                                                       x2_zoomed + handle_size / 2, y1_zoomed + handle_size / 2,
                                                       fill="blue", tags="bbox")
                handle3 = self.canvas.create_rectangle(x2_zoomed - handle_size / 2, y2_zoomed - handle_size / 2,
                                                       x2_zoomed + handle_size / 2, y2_zoomed + handle_size / 2,
                                                       fill="blue", tags="bbox")
                handle4 = self.canvas.create_rectangle(x1_zoomed - handle_size / 2, y2_zoomed - handle_size / 2,
                                                       x1_zoomed + handle_size / 2, y2_zoomed + handle_size / 2,
                                                       fill="blue", tags="bbox")

                self.bbox_handles.extend([handle1, handle2, handle3, handle4])
            else:
                pass

    def on_mouse_move(self, event):
        x, y = event.x, event.y

        if self.highlighted_bbox:
            self.canvas.delete(self.highlighted_bbox)
            self.highlighted_bbox = None

        for idx, (x1, y1, x2, y2) in enumerate(self.bbox_coords):
            # On récupère la couleur de la box associée
            category_name = None
            if idx < len(self.bbox_files):
                bbox_file = self.bbox_files[idx]
                # Chercher la catégorie associée à ce fichier
                for box in self.image_manager.get_bounding_boxes():
                    if box[2] == bbox_file:
                        category_name = box[1]
                        break
            color = "red" if category_name and category_name.lower() == "unknown" else "green"
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.highlighted_bbox = self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, fill=color, stipple="gray25", width=2)
                break
        self.root.title(f"{self.image_manager.get_image_path()} (x:{x}, y:{y})")

    def on_right_click(self, event):
        x, y = event.x, event.y

        for idx, (x1, y1, x2, y2) in enumerate(self.bbox_coords):
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.dragging_bbox = idx
                self.drag_offset_x = x - x1
                self.drag_offset_y = y - y1
                break

    def on_right_drag(self, event):
        if self.dragging_bbox is not None:
            x1, y1, x2, y2 = self.bbox_coords[self.dragging_bbox]
            delta_x = event.x - self.drag_offset_x - x1
            delta_y = event.y - self.drag_offset_y - y1

            self.canvas.move(self.bboxes[self.dragging_bbox], delta_x, delta_y)
            
            # Déplacer aussi le label de catégorie
            if self.dragging_bbox < len(self.bbox_labels):
                self.canvas.move(self.bbox_labels[self.dragging_bbox], delta_x, delta_y)

            self.bbox_coords[self.dragging_bbox] = (x1 + delta_x, y1 + delta_y, x2 + delta_x, y2 + delta_y)

            self.update_handles(self.dragging_bbox)

    def on_right_release(self, event):
        self.dragging_bbox = None

    def delete_bbox(self):
        x, y = self.root.winfo_pointerxy()
        x -= self.root.winfo_rootx()
        y -= self.root.winfo_rooty()

        for idx, (x1, y1, x2, y2) in enumerate(self.bbox_coords):
            if x1 <= x <= x2 and y1 <= y <= y2:
                bbox_file_to_delete = self.bbox_files[idx]
                self.deleted_bboxes.add(bbox_file_to_delete)
                self.draw_bounding_boxes()  # Couleur mise à jour automatiquement
                break

    def edit_category_name(self):
        x, y = self.root.winfo_pointerxy()
        x -= self.root.winfo_rootx()
        y -= self.root.winfo_rooty()

        for idx, (x1, y1, x2, y2) in enumerate(self.bbox_coords):
            if x1 <= x <= x2 and y1 <= y <= y2:
                base_name = os.path.splitext(self.image_manager.image_list[self.image_manager.current_image_index])[0]
                bbox_files = [f for f in os.listdir(self.image_manager.root_folder) if f.startswith(base_name) and f.endswith('.json')]

                if self.bbox_files[idx] in bbox_files:
                    bbox_file = self.bbox_files[idx]
                    bbox_file_path = os.path.join(self.image_manager.root_folder, bbox_file)

                    with open(bbox_file_path, 'r') as f:
                        data = json.load(f)

                    new_category_name = simpledialog.askstring("Edit Category", "Enter new category name:", initialvalue=data['annotation']['category_name'])

                    if new_category_name:
                        data['annotation']['category_name'] = new_category_name

                        # Sauvegarder avec contrôle de cohérence
                        self.save_json_with_consistency_check(bbox_file_path, data)

                        self.draw_bounding_boxes()  # Couleur mise à jour automatiquement
                break

    def fill_box_full_image(self):
        # Si une seule boxe affichée (non supprimée)
        current_boxes = self.image_manager.get_bounding_boxes()
        boxes = [(i, box) for i, box in enumerate(current_boxes) if box[2] not in self.deleted_bboxes]
        print("OK fill_box_full_image")
        if len(boxes) == 1:
            print("resize")
            idx, (bbox, category_name, bbox_file) = boxes[0]
            # Trouver l'index correct dans self.bbox_coords
            bbox_coords_idx = None
            for i, (_, _, file) in enumerate(current_boxes):
                if file == bbox_file and file not in self.deleted_bboxes:
                    bbox_coords_idx = i
                    break
            
            if bbox_coords_idx is not None:
                # Mettre à jour la box pour qu'elle englobe toute l'image
                width, height = self.image.width, self.image.height
                # Mettre à jour les coordonnées zoomées
                x1_zoomed = 0
                y1_zoomed = 0
                x2_zoomed = width
                y2_zoomed = height
                self.bbox_coords[bbox_coords_idx] = (x1_zoomed, y1_zoomed, x2_zoomed, y2_zoomed)
                print(self.bbox_coords[bbox_coords_idx])
                
                # Sauvegarder les nouvelles coordonnées dans le fichier JSON
                bbox_file_path = os.path.join(self.image_manager.root_folder, bbox_file)
                with open(bbox_file_path, 'r') as f:
                    data = json.load(f)
                
                # Convertir les coordonnées zoomées en coordonnées originales
                x1_original = x1_zoomed / self.image_manager.zoom_factor
                y1_original = y1_zoomed / self.image_manager.zoom_factor
                width_original = (x2_zoomed - x1_zoomed) / self.image_manager.zoom_factor
                height_original = (y2_zoomed - y1_zoomed) / self.image_manager.zoom_factor
                
                data['annotation']['bbox'] = [x1_original, y1_original, width_original, height_original]
                
                # Sauvegarder avec contrôle de cohérence
                self.save_json_with_consistency_check(bbox_file_path, data)
                
                self.draw_bounding_boxes()

    def fill_box_full_width(self):
        # Si une seule boxe affichée (non supprimée)
        current_boxes = self.image_manager.get_bounding_boxes()
        print("AVANT fill_box_full_width")
        boxes = [(i, box) for i, box in enumerate(current_boxes) if box[2] not in self.deleted_bboxes]
        print("OK fill_box_full_width")
        if len(boxes) == 1:
            print("resize width")
            idx, (bbox, category_name, bbox_file) = boxes[0]
            # Trouver l'index correct dans self.bbox_coords
            bbox_coords_idx = None
            for i, (_, _, file) in enumerate(current_boxes):
                if file == bbox_file and file not in self.deleted_bboxes:
                    bbox_coords_idx = i
                    break
            
            if bbox_coords_idx is not None:
                # Mettre à jour la box pour qu'elle prenne toute la largeur en conservant sa position verticale
                width, height = self.image.width, self.image.height
                x1, y1, w, h = bbox
                # Mettre à jour les coordonnées zoomées
                x1_zoomed = 0  # Commence à x=0
                y1_zoomed = y1 * self.image_manager.zoom_factor  # Conserve la position Y
                x2_zoomed = width  # Prend toute la largeur
                y2_zoomed = (y1 + h) * self.image_manager.zoom_factor  # Conserve la hauteur
                self.bbox_coords[bbox_coords_idx] = (x1_zoomed, y1_zoomed, x2_zoomed, y2_zoomed)
                print(self.bbox_coords[bbox_coords_idx])
                
                # Sauvegarder les nouvelles coordonnées dans le fichier JSON
                bbox_file_path = os.path.join(self.image_manager.root_folder, bbox_file)
                with open(bbox_file_path, 'r') as f:
                    data = json.load(f)
                
                # Convertir les coordonnées zoomées en coordonnées originales
                x1_original = x1_zoomed / self.image_manager.zoom_factor
                y1_original = y1_zoomed / self.image_manager.zoom_factor
                width_original = (x2_zoomed - x1_zoomed) / self.image_manager.zoom_factor
                height_original = (y2_zoomed - y1_zoomed) / self.image_manager.zoom_factor
                
                data['annotation']['bbox'] = [x1_original, y1_original, width_original, height_original]
                
                # Sauvegarder avec contrôle de cohérence
                self.save_json_with_consistency_check(bbox_file_path, data)
                
                self.draw_bounding_boxes()

    def image_save_folder(self):
        # Croper automatiquement si activé
        if self.auto_crop and self.crop_height > 0:
            print("🔄 AUTO_CROP activé - Crop automatique avant sauvegarde")
            self.crop_bottom_zone()
        
        origin_image_path = self.image_manager.get_image_path()
        base_name = os.path.splitext(self.image_manager.image_list[self.image_manager.current_image_index])[0]

        target_image_path = os.path.join(self.image_manager.root_folder, self.image_manager.save_folder, os.path.basename(origin_image_path))
        os.rename(origin_image_path, target_image_path)

        bbox_files = [f for f in os.listdir(self.image_manager.root_folder) if f.startswith(base_name) and f.endswith('.json')]

        # Créer une correspondance entre les noms de fichiers et leurs coordonnées
        bbox_coords_map = {}
        for idx, bbox_file in enumerate(self.bbox_files):
            if idx < len(self.bbox_coords):
                bbox_coords_map[bbox_file] = self.bbox_coords[idx]

        for bbox_file in bbox_files:
            if bbox_file not in self.deleted_bboxes:
                bbox_file_path = os.path.join(self.image_manager.root_folder, bbox_file)

                with open(bbox_file_path, 'r') as f:
                    data = json.load(f)

                # Utiliser les coordonnées mises à jour si disponibles
                if bbox_file in bbox_coords_map:
                    x1, y1, x2, y2 = bbox_coords_map[bbox_file]
                    x1_original = x1 / self.image_manager.zoom_factor
                    y1_original = y1 / self.image_manager.zoom_factor
                    x2_original = x2 / self.image_manager.zoom_factor
                    y2_original = y2 / self.image_manager.zoom_factor

                    data['annotation']['bbox'] = [x1_original, y1_original, x2_original - x1_original, y2_original - y1_original]

                target_bbox_path = os.path.join(self.image_manager.root_folder, self.image_manager.save_folder, bbox_file)
                # Sauvegarder avec contrôle de cohérence
                self.save_json_with_consistency_check(target_bbox_path, data)

                os.remove(bbox_file_path)
            else:
                bbox_file_path = os.path.join(self.image_manager.root_folder, bbox_file)
                os.remove(bbox_file_path)

        # Seulement passer à l'image suivante si on n'est pas en auto-traitement
        # (pour éviter les conflits de thread)
        if not self.auto_processing:
            # Recharger la liste des images après sauvegarde (mode manuel)
            self.image_manager.reload_images()
            
            # Ajuster l'index si nécessaire (si on était sur la dernière image)
            if self.image_manager.current_image_index >= len(self.image_manager.image_list):
                self.image_manager.current_image_index = len(self.image_manager.image_list) - 1
            
            self.show_next_image(None)
        else:
            # En auto-traitement, ne pas recharger la liste pour éviter les problèmes d'index
            # L'auto-traitement gère lui-même l'index
            pass

    def image_save_tmp(self):
        """Sauvegarde les boîtes de l'image courante sans déplacer l'image ni passer à la suivante."""
        origin_image_path = self.image_manager.get_image_path()
        base_name = os.path.splitext(self.image_manager.image_list[self.image_manager.current_image_index])[0]

        # Créer une correspondance entre les noms de fichiers et leurs coordonnées
        bbox_coords_map = {}
        for idx, bbox_file in enumerate(self.bbox_files):
            if idx < len(self.bbox_coords):
                bbox_coords_map[bbox_file] = self.bbox_coords[idx]

        bbox_files = [f for f in os.listdir(self.image_manager.root_folder) if f.startswith(base_name) and f.endswith('.json')]

        for bbox_file in bbox_files:
            if bbox_file not in self.deleted_bboxes:
                bbox_file_path = os.path.join(self.image_manager.root_folder, bbox_file)

                with open(bbox_file_path, 'r') as f:
                    data = json.load(f)

                # Utiliser les coordonnées mises à jour si disponibles
                if bbox_file in bbox_coords_map:
                    x1, y1, x2, y2 = bbox_coords_map[bbox_file]
                    x1_original = x1 / self.image_manager.zoom_factor
                    y1_original = y1 / self.image_manager.zoom_factor
                    x2_original = x2 / self.image_manager.zoom_factor
                    y2_original = y2 / self.image_manager.zoom_factor

                    data['annotation']['bbox'] = [x1_original, y1_original, x2_original - x1_original, y2_original - y1_original]

                # Sauvegarder avec contrôle de cohérence
                self.save_json_with_consistency_check(bbox_file_path, data)

    def draw_crop_zone(self):
        """Dessine un carré vert représentant la zone qui sera croppée"""
        if self.crop_height > 0 and self.original_image:
            # Calculer les coordonnées de la zone de crop
            image_width = self.original_image.width
            image_height = self.original_image.height
            
            # Zone de crop en bas de l'image
            x1 = 0
            y1 = image_height - self.crop_height
            x2 = image_width
            y2 = image_height
            
            # Appliquer le zoom
            x1_zoomed = x1 * self.image_manager.zoom_factor
            y1_zoomed = y1 * self.image_manager.zoom_factor
            x2_zoomed = x2 * self.image_manager.zoom_factor
            y2_zoomed = y2 * self.image_manager.zoom_factor
            
            # Dessiner le rectangle vert semi-transparent
            self.canvas.create_rectangle(
                x1_zoomed, y1_zoomed, x2_zoomed, y2_zoomed,
                outline="green", fill="green", stipple="gray25", width=2,
                tags="crop_zone"
            )
            
            # Ajouter un label
            self.canvas.create_text(
                (x1_zoomed + x2_zoomed) / 2, (y1_zoomed + y2_zoomed) / 2,
                text=f"CROP ZONE ({self.crop_height}px)", 
                fill="white", font=("Arial", 12, "bold"),
                tags="crop_zone"
            )

    def crop_bottom_zone(self):
        """Croppe la zone du bas de l'image et met à jour tous les JSON associés"""
        if self.crop_height <= 0:
            print("CROP_HEIGHT non défini ou invalide")
            return
            
        if not self.original_image:
            print("Aucune image chargée")
            return
            
        # Vérifier que l'image est assez haute pour être croppée
        if self.original_image.height <= self.crop_height:
            print(f"Image trop petite pour être croppée ({self.original_image.height}px <= {self.crop_height}px)")
            return
            
        try:
            # 1. Cropper l'image
            original_width = self.original_image.width
            original_height = self.original_image.height
            new_height = original_height - self.crop_height
            
            # Créer la zone de crop (x, y, x+width, y+height)
            crop_box = (0, 0, original_width, new_height)
            cropped_image = self.original_image.crop(crop_box)
            
            # Sauvegarder l'image croppée
            image_path = self.image_manager.get_image_path()
            cropped_image.save(image_path)
            print(f"Image croppée: {original_width}x{original_height} → {original_width}x{new_height}")
            
            # 2. Mettre à jour tous les JSON associés
            base_name = os.path.splitext(self.image_manager.image_list[self.image_manager.current_image_index])[0]
            bbox_files = [f for f in os.listdir(self.image_manager.root_folder) 
                         if f.startswith(base_name) and f.endswith('.json')]
            
            updated_files = 0
            for bbox_file in bbox_files:
                bbox_file_path = os.path.join(self.image_manager.root_folder, bbox_file)
                
                try:
                    with open(bbox_file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Mettre à jour la hauteur de l'image
                    if 'image' in data:
                        data['image']['height'] = new_height
                    
                    # Sauvegarder avec contrôle de cohérence
                    self.save_json_with_consistency_check(bbox_file_path, data)
                    updated_files += 1
                    
                except Exception as e:
                    print(f"Erreur lors de la mise à jour de {bbox_file}: {e}")
            
            print(f"Mis à jour {updated_files} fichiers JSON")
            
            # 3. Recharger l'image et redessiner
            self.load_image()
            self.draw_bounding_boxes()
            self.draw_crop_zone()
            
        except Exception as e:
            print(f"Erreur lors du crop: {e}")

    def start_auto_processing(self, event=None):
        """Démarre l'auto-traitement de toutes les images"""
        if self.auto_processing:
            print("Auto-traitement déjà en cours...")
            return
            
        # Confirmation avant de démarrer
        import tkinter.messagebox as msgbox
        total_images = len(self.image_manager.image_list)
        remaining_images = total_images - self.image_manager.current_image_index
        
        if msgbox.askyesno("Auto-traitement", 
                          f"Traiter automatiquement {remaining_images} images restantes ?\n\n"
                          f"Appuyez sur ESC pour arrêter à tout moment."):
            self.auto_processing = True
            self.update_title_with_progress()
            self.auto_process_all()
    
    def stop_auto_processing(self, event=None):
        """Arrête l'auto-traitement"""
        if self.auto_processing:
            self.auto_processing = False
            print("Auto-traitement arrêté par l'utilisateur")
            self.update_title_with_progress()
    
    def auto_process_all(self):
        """Traite automatiquement toutes les images restantes"""
        import threading
        import time
        
        def process_loop():
            while self.auto_processing:
                try:
                    # Vérifier qu'il reste des images à traiter
                    if self.image_manager.current_image_index >= len(self.image_manager.image_list):
                        self.auto_processing = False
                        print("Auto-traitement terminé - Toutes les images ont été traitées")
                        break
                    
                    print(f"Auto-traitement: {self.image_manager.current_image_index + 1}/{len(self.image_manager.image_list)}")
                    
                    # Vérifier que l'image courante existe encore (au cas où elle aurait été déplacée)
                    current_image_path = self.image_manager.get_image_path()
                    if not os.path.exists(current_image_path):
                        print(f"Image {current_image_path} n'existe plus, passage à la suivante")
                        self.image_manager.current_image_index += 1
                        if self.image_manager.current_image_index >= len(self.image_manager.image_list):
                            self.auto_processing = False
                            print("Auto-traitement terminé - Toutes les images ont été traitées")
                            break
                        continue
                    
                    # Sauvegarder l'image courante (équivalent à touche 's')
                    self.image_save_folder()
                    
                    # Petite pause pour éviter les bugs
                    time.sleep(0.5)
                    
                    # Passer à l'image suivante si elle existe
                    if self.image_manager.current_image_index < len(self.image_manager.image_list) - 1:
                        self.image_manager.current_image_index += 1
                        # Recharger l'image suivante
                        self.root.after(0, self._load_next_image_for_auto_processing)
                        time.sleep(0.3)
                    else:
                        # Fin du répertoire
                        self.auto_processing = False
                        print("Auto-traitement terminé - Toutes les images ont été traitées")
                        break
                        
                except Exception as e:
                    print(f"Erreur lors de l'auto-traitement: {e}")
                    self.auto_processing = False
                    break
            
            # Mettre à jour le titre à la fin
            self.root.after(0, self.update_title_with_progress)
        
        # Lancer dans un thread séparé pour ne pas bloquer l'interface
        self.auto_process_thread = threading.Thread(target=process_loop, daemon=True)
        self.auto_process_thread.start()
    
    def _reload_current_image(self):
        """Helper pour recharger l'image courante pendant l'auto-traitement"""
        try:
            self.drag_start_x = 0
            self.drag_start_y = 0
            self.image_manager.zoom_factor = 1
            self.deleted_bboxes = set()
            self.load_image()
            self.update_title_with_progress()
            self.draw_bounding_boxes()
            self.draw_crop_zone()
            self.update_window_color_from_current_image()
        except Exception as e:
            print(f"Erreur lors du rechargement de l'image: {e}")
            self.auto_processing = False
    
    def _load_next_image_for_auto_processing(self):
        """Helper pour recharger l'image suivante pendant l'auto-traitement"""
        try:
            self.drag_start_x = 0
            self.drag_start_y = 0
            self.image_manager.zoom_factor = 1
            self.deleted_bboxes = set()
            self.load_image()
            self.update_title_with_progress()
            self.draw_bounding_boxes()
            self.draw_crop_zone()
            self.update_window_color_from_current_image()
        except Exception as e:
            print(f"Erreur lors du rechargement de l'image: {e}")
            self.auto_processing = False
    
    def update_title_with_progress(self):
        """Met à jour le titre avec la progression de l'auto-traitement"""
        current = self.image_manager.current_image_index + 1
        total = len(self.image_manager.image_list)
        
        if self.auto_processing:
            progress = f" [AUTO-TRAITEMENT: {current}/{total}]"
        else:
            progress = f" [{current}/{total}]"
        
        # Mettre à jour le titre
        current_title = self.root.title()
        if " | " in current_title:
            base_title = current_title.split(" | ")[0]
        else:
            base_title = current_title
        
        new_title = f"{base_title}{progress}"
        self.root.title(new_title)

    def quit_app(self):
        self.root.destroy()


CATEGORY_NAME = 'unknown'
CROP_HEIGHT = 0  # Hauteur en pixels à croper (0 = désactivé)
AUTO_CROP = False  # Croper automatiquement lors de la sauvegarde (touche 's')

if __name__ == "__main__":
    root_folder = "/home/aobled/Downloads/tmp_multi"
    viewer = PhotoViewer(root_folder, category_name=CATEGORY_NAME, crop_height=CROP_HEIGHT, auto_crop=AUTO_CROP)

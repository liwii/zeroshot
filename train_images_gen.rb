IMAGES_DIR = "JPEGImages_128x128"

f = File.open( "trainclasses.txt")
train_classes = f.read.split("\n")
f.close

train_classes.each do |c|
    Dir.entries(File.join(IMAGES_DIR, c)).each do |file|
        next if file == ".." || file == "."
        puts("#{File.join(c, file)} #{c}")
    end
end